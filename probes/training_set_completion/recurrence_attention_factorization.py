"""CPU-only row-mass/within-row factorization of the accepted val7511 NO cache effect."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding


SOURCE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase")
ATTEMPT = SOURCE / "attempt-003"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-phase-transfer/attention-factorization")
DEST = slice(2112, 2121)
ATOL = 2e-4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def factorize(p_row, o_native, v_row, delta):
    """Return four outputs at fixed outside-row state, in float64."""
    p, o, v, delta = (x.double() for x in (p_row, o_native, v_row, delta))
    require(p.ndim == delta.ndim == 3 and p.shape == delta.shape and
            o.shape[:2] == p.shape[:2] and v.shape == (p.shape[0], p.shape[2], o.shape[2]),
            "row factorization tensor shapes differ")
    require(bool(torch.isfinite(p).all() and torch.isfinite(o).all() and
                 torch.isfinite(v).all() and torch.isfinite(delta).all() and (p >= 0).all()),
            "nonfinite or negative factorization input")
    m = p.sum(-1, keepdim=True)
    # At exactly zero or one mass, a conditional row or outside value is unidentified.
    require(bool(((m > 1e-12) & (m < 1 - 1e-12)).all()), "row/outside mass is not identifiable")
    u = p / m
    logw = torch.where(p > 0, p.log(), -torch.inf) + delta
    log_row = torch.logsumexp(logw, dim=-1, keepdim=True)
    log_total = torch.logaddexp(log_row, torch.log1p(-m))
    mp = torch.exp(log_row - log_total)
    up = torch.exp(logw - log_row)
    require(bool(torch.isfinite(mp).all() and torch.isfinite(up).all()), "reweighted mass is nonfinite")
    v_native = torch.matmul(u, v)
    v_phase = torch.matmul(up, v)
    v_outside = (o - m * v_native) / (1 - m)
    outputs = {
        "m_u": m * v_native + (1 - m) * v_outside,
        "mp_u": mp * v_native + (1 - mp) * v_outside,
        "m_up": m * v_phase + (1 - m) * v_outside,
        "mp_up": mp * v_phase + (1 - mp) * v_outside,
    }
    effects = {
        "mass": outputs["mp_u"] - outputs["m_u"],
        "allocation": outputs["m_up"] - outputs["m_u"],
        "interaction": outputs["mp_up"] - outputs["mp_u"] - outputs["m_up"] + outputs["m_u"],
        "total": outputs["mp_up"] - outputs["m_u"],
    }
    return {"m": m, "mp": mp, "u": u, "up": up, "log_denominator": log_total,
            "outputs": outputs, "effects": effects}


def selfcheck():
    torch.manual_seed(22)
    q, k, v = torch.randn(2, 3, 4, dtype=torch.float64), torch.randn(2, 5, 4, dtype=torch.float64), torch.randn(2, 5, 4, dtype=torch.float64)
    scores = q @ k.transpose(-1, -2)
    p = torch.softmax(scores, dim=-1)
    delta = torch.randn(2, 3, 2, dtype=torch.float64) / 3
    result = factorize(p[:, :, 1:3], p @ v, v[:, 1:3], delta)
    changed = scores.clone()
    changed[:, :, 1:3] += delta
    brute = torch.softmax(changed, dim=-1) @ v
    require(torch.allclose(result["outputs"]["m_u"], p @ v, atol=1e-12, rtol=0) and
            torch.allclose(result["outputs"]["mp_up"], brute, atol=1e-12, rtol=0) and
            torch.allclose(sum((result["effects"][x] for x in ("mass", "allocation", "interaction"))),
                           result["effects"]["total"], atol=1e-12, rtol=0),
            "synthetic identity or brute-force endpoint failed")
    require(float((factorize(p[:, :, 1:3], p @ v, v[:, 1:3], -delta)["outputs"]["mp_up"] - brute).abs().max()) > 1e-3,
            "wrong reweighting sign escaped selfcheck")
    for mass in (1e-9, 1 - 1e-9):
        row = torch.full((1, 1, 2), mass / 2, dtype=torch.float64)
        outside = torch.tensor([[[2.0, -1.0]]], dtype=torch.float64)
        row_v = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
        native = mass * row_v.mean(1, keepdim=True) + (1 - mass) * outside
        edge = factorize(row, native, row_v, torch.zeros_like(row))
        require(bool(torch.isfinite(edge["outputs"]["mp_up"]).all()) and
                torch.allclose(edge["outputs"]["mp_up"], native, atol=1e-12, rtol=0),
                "near-boundary mass became unstable")
    for mass in (0.0, 1.0):
        try:
            factorize(torch.full((1, 1, 2), mass / 2), torch.zeros(1, 1, 2), torch.zeros(1, 2, 2), torch.zeros(1, 1, 2))
        except ValueError:
            pass
        else:
            raise AssertionError("unidentified boundary mass was admitted")


def metrics(effects, reference, reconstruction_error, *, final):
    def rows(x):
        x = x[:, -1] if final else x
        return x.reshape(x.shape[0], -1)
    target = rows(reference)
    noise = torch.linalg.vector_norm(rows(reconstruction_error), dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    report = {"reference_norm": target_norm.tolist(), "reconstruction_error_norm": noise.tolist()}
    for name, value in effects.items():
        z = rows(value)
        norm = torch.linalg.vector_norm(z, dim=-1)
        denominator = norm * target_norm
        cosine = (z * target).sum(-1) / denominator.clamp_min(1e-30)
        report[name] = {"norm": norm.tolist(),
                        "alignment_to_accepted_fixed_delta": [float(c) if float(d) > 1e-12 else None
                                                               for c, d in zip(cosine, denominator, strict=True)],
                        "norm_to_reconstruction_error": [float(n / e) if float(e) > 1e-12 else None
                                                         for n, e in zip(norm, noise, strict=True)]}
    return report


def run():
    selfcheck()
    acceptance = SOURCE / "lead-acceptance.json"
    result_path = ATTEMPT / "result.json"
    receipt_path = ATTEMPT / "receipt.json"
    accepted, result, receipt = (json.loads(p.read_text()) for p in (acceptance, result_path, receipt_path))
    require(accepted["status"] == "lead-accepted" and "val7511 row89 x2" in accepted["model_scope"] and
            result["status"] == "candidate" and result["intermediate"]["status"] == "candidate" and
            receipt["status"] == "candidate_complete" and receipt["result"] == literal_binding(result_path),
            "accepted source identity/status changed")
    paths = {"NN": ATTEMPT / "NN-intermediates.pt", "phase": ATTEMPT / "phase-and-prekeys.pt",
             "fixed": ATTEMPT / "fixed-native-decomposition.pt"}
    expected = {"NN": result["cells"]["NN"]["intermediates"], "phase": result["phase_and_prekeys"],
                "fixed": result["decomposition"]}
    require(all(literal_binding(path) == expected[name] for name, path in paths.items()), "source tensor binding changed")
    nn, phase, fixed = (torch.load(paths[name], map_location="cpu", weights_only=True) for name in paths)
    require(len(nn["layers"]) == len(phase["layers"]) == len(fixed["NO"]) == 28, "layer denominator changed")
    stacks = {name: [] for name in ("m", "mp", "u", "up", "log_denominator")}
    for name in ("m_u", "mp_u", "m_up", "mp_up", "mass", "allocation", "interaction", "total"):
        stacks[name] = []
    per_layer = []
    endpoint_error = brute_error = denominator_error = identity_error = 0.0
    for i, (n, ph, f) in enumerate(zip(nn["layers"], phase["layers"], fixed["NO"], strict=True)):
        require(n["attention_probabilities_reconstructed"].shape == (16, 6, 2127) and
                n["q_post_reconstructed"].shape == n["reconstructed_headout"].shape == (16, 6, 128) and
                n["native_V_destination"].shape == (8, 9, 128) and
                torch.equal(n["native_V_destination"], ph["native_V_dest"]),
                "NN query/attention/native V identity changed")
        keys = ph["candidate_keys"]
        q = n["q_post_reconstructed"].double()
        delta = q @ (keys["NO"].double() - keys["NN"].double()).repeat_interleave(2, 0).transpose(-1, -2) * (128 ** -0.5)
        p = n["attention_probabilities_reconstructed"][:, :, DEST]
        v = n["native_V_destination"].repeat_interleave(2, 0)
        x = factorize(p, n["reconstructed_headout"], v, delta)
        endpoint_error = max(endpoint_error, float((x["outputs"]["mp_up"] - f["fixed_native_headout"].double()).abs().max()))
        brute_error = max(brute_error, float((x["outputs"]["mp_up"] - n["fixed_native_brute_headout"]["NO"].double()).abs().max()))
        denominator_error = max(denominator_error, float((x["log_denominator"].exp() - f["denominator"].double()).abs().max()))
        identity_error = max(identity_error, float((x["effects"]["mass"] + x["effects"]["allocation"] +
                                                   x["effects"]["interaction"] - x["effects"]["total"]).abs().max()))
        for name in ("m", "mp", "u", "up", "log_denominator"):
            stacks[name].append(x[name])
        for name in x["outputs"]:
            stacks[name].append(x["outputs"][name])
        for name in x["effects"]:
            stacks[name].append(x["effects"][name])
        reference = f["fixed_native_headout"].double() - n["actual_headout"].double()
        noise = n["actual_headout"].double() - n["reconstructed_headout"].double()
        per_layer.append({"layer": i,
                          "last_S": metrics(x["effects"], reference, noise, final=True),
                          "all_S": metrics(x["effects"], reference, noise, final=False),
                          "native_row_mass_range": [float(x["m"].min()), float(x["m"].max())],
                          "NO_row_mass_range": [float(x["mp"].min()), float(x["mp"].max())]})
    require(endpoint_error <= ATOL and brute_error <= ATOL and identity_error <= 1e-10,
            f"factorization failed accepted/brute endpoint or identity: {endpoint_error}, {brute_error}, {identity_error}")
    tensor = {name: torch.stack(values) for name, values in stacks.items()}
    require(all(x.shape[:3] == (28, 16, 6) for x in tensor.values()), "not all layers/heads/S queries retained")
    OUT.mkdir(parents=True, exist_ok=False)
    tensor_path = OUT / "factorization.pt"
    torch.save(tensor, tensor_path)
    summary = {"schema": "recurrence_attention_factorization.v1", "status": "candidate", "scope": "val7511 row89 x2; fixed NN Q and all outside-row NN K/V; NO destination K",
               "source": {"acceptance": literal_binding(acceptance), "result": literal_binding(result_path),
                          "receipt": literal_binding(receipt_path), "producer": literal_binding(Path(__file__)),
                          **{name: literal_binding(path) for name, path in paths.items()}},
               "tensor": literal_binding(tensor_path), "denominator": {"layers": 28, "heads": 16, "S_queries": 6, "row_tokens": 9},
               "qualification": {"endpoint_vs_accepted_NO_max_abs": endpoint_error,
                                 "endpoint_vs_independent_brute_max_abs": brute_error,
                                 "saved_denominator_max_abs": denominator_error,
                                 "factorial_identity_max_abs": identity_error,
                                 "accepted_NN_attention_reconstruction_max_abs": result["intermediate"]["attention_output_max_abs"],
                                 "atol": ATOL},
               "mass_range": {name: [float(tensor[name].min()), float(tensor[name].max())] for name in ("m", "mp")},
               "per_layer": per_layer,
               "interpretation_boundary": "Each head/layer is evaluated at fixed NN state; no cross-layer causal sum."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "candidate", "summary": str(OUT / "summary.json"),
                      "endpoint_max_abs": endpoint_error, "brute_max_abs": brute_error}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        print("PASS recurrence attention factorization selfcheck")
    else:
        run()
