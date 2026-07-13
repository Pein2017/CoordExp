## ADDED Requirements

### Requirement: Open-assistant continuation
Inference prompt construction SHALL accept an optional immutable assistant
continuation whose operational meaning is canonical text appended directly
inside the assistant turn already opened by the Qwen generation prompt. The
continuation MUST NOT be represented as a completed assistant message, a new
chat turn, or an independently tokenized suffix.

When a continuation is present, the prompt builder MUST construct the ordinary
open-assistant chat text first and locate the final assistant-content start. The
interval from that content start to the continuation start MUST contain no
assistant terminator, end-of-sequence token, new chat-turn opener, or
pre-existing assistant content; terminators belonging to earlier completed
system or user turns remain valid. The builder MUST append the continuation with
no inserted separator or terminal token and tokenize the complete combined text
with automatic special-token insertion disabled. It MUST reject empty
continuation text, image placeholders, assistant terminators, end-of-sequence
tokens, chat-turn openers, extra turns, tokenization truncation, or a combined
prompt that does not contain exactly one image placeholder. When no continuation
is present, ordinary prompt text and token identifiers MUST remain unchanged.

#### Scenario: Valid continuation inside open assistant turn
- **WHEN** canonical compact object rows are supplied as an assistant
  continuation
- **THEN** they appear immediately after the ordinary open-assistant generation
  prompt with no new chat header, separator, assistant terminator, or
  end-of-sequence token
- **AND** the backend receives token identifiers from tokenizing the complete
  combined prompt

#### Scenario: Completed assistant message rejected
- **WHEN** continuation construction would add a completed assistant message or
  another conversation turn
- **THEN** prompt validation fails before image materialization or generation

#### Scenario: Forbidden control token rejected
- **WHEN** continuation text contains an image placeholder, assistant
  terminator, end-of-sequence token, or chat-turn opener
- **THEN** prompt validation fails and identifies the forbidden boundary class

#### Scenario: No-continuation compatibility
- **WHEN** no assistant continuation is supplied
- **THEN** prompt text and prompt token identifiers are byte-for-byte and
  identifier-for-identifier equal to the existing ordinary prompt path

### Requirement: Assistant-continuation prompt evidence
Every prompt record built with an assistant continuation SHALL record the full
combined prompt text and token identifiers, full-prompt fingerprint,
continuation text hash, final assistant-content start, continuation byte and
character spans, and continuation token impact span. The existing `prompt_text`
field MUST retain its authored task-prompt meaning; exact ordinary or continued
chat text MUST remain in `chat_text` and continued prompt evidence MUST serialize
it explicitly as `full_chat_text`. The token impact span MUST begin at the
longest common token prefix between the ordinary prompt and the fully retokenized
combined prompt and MUST end at the combined prompt token count. It MUST NOT be
described as an independently tokenized continuation span.

Backend prompt-parity validation MUST compare the full combined prompt token
identifiers. Evidence MUST prove that the full prompt contains exactly one image
placeholder and that the interval from the final assistant-content start to the
continuation boundary contains no assistant terminator, end-of-sequence token,
new turn opener, or pre-existing assistant content.

#### Scenario: Earlier completed turn terminator remains valid
- **WHEN** an ordinary system or user turn ends with its required chat terminator
  before the final open assistant turn
- **THEN** continuation validation accepts that earlier completed-turn boundary

#### Scenario: Terminator inside open assistant content rejected
- **WHEN** an assistant terminator appears after the final assistant-content
  start and before the continuation boundary
- **THEN** prompt validation fails before generation

#### Scenario: Boundary retokenization changes a prior token
- **WHEN** full-prompt retokenization changes the final ordinary-prompt token at
  the continuation boundary
- **THEN** the recorded token impact span begins at that changed token
- **AND** the full combined token identifiers remain the parity authority

#### Scenario: Exact continuation evidence
- **WHEN** a continued prompt is materialized
- **THEN** its byte span reconstructs the exact continuation bytes from the full
  prompt
- **AND** its prompt fingerprint and token impact span are present in the prompt
  record

#### Scenario: Backend prompt mismatch
- **WHEN** backend prompt token identifiers differ from the recorded full
  combined prompt identifiers
- **THEN** inference fails before accepting the decode result
