export class ApiError extends Error {
  constructor(message, { status = 0, body = null, cause = undefined } = {}) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

export function createApiClient({
  fetchImpl = globalThis.fetch?.bind(globalThis),
  origin = globalThis.location?.origin,
} = {}) {
  if (typeof fetchImpl !== 'function') throw new TypeError('fetchImpl must be a function');
  if (typeof origin !== 'string' || !origin) throw new TypeError('origin is required');
  let csrfToken = '';

  const requestJson = async (path, { method = 'GET', body, mutation = false } = {}) => {
    const url = sameOriginUrl(path, origin);
    if (mutation && !csrfToken) throw new ApiError('local session is not initialized');
    const headers = { Accept: 'application/json' };
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (mutation) headers['x-csrf-token'] = csrfToken;
    let response;
    try {
      response = await fetchImpl(url.pathname + url.search, {
        method,
        credentials: 'same-origin',
        cache: 'no-store',
        headers,
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
    } catch (error) {
      throw new ApiError('request did not receive a response', { cause: error });
    }
    const responseBody = await response.json().catch(() => null);
    if (!response.ok) {
      throw new ApiError(
        responseBody?.error?.message || `Request failed (${response.status})`,
        { status: response.status, body: responseBody },
      );
    }
    return responseBody;
  };

  return Object.freeze({
    async bootstrapSession() {
      const body = await requestJson('/api/session');
      if (typeof body?.csrf_token !== 'string' || !body.csrf_token) {
        throw new ApiError('session response lacks a CSRF token');
      }
      csrfToken = body.csrf_token;
      return body;
    },
    getJson(path) { return requestJson(path); },
    postJson(path, body) { return requestJson(path, { method: 'POST', body, mutation: true }); },
    putJson(path, body) { return requestJson(path, { method: 'PUT', body, mutation: true }); },
    deleteJson(path) { return requestJson(path, { method: 'DELETE', mutation: true }); },
  });
}

function sameOriginUrl(path, origin) {
  if (typeof path !== 'string' || !path.startsWith('/') || path.startsWith('//')) {
    throw new TypeError('API path must be root-relative');
  }
  const url = new URL(path, origin);
  if (url.origin !== origin || !url.pathname.startsWith('/api/')) {
    throw new TypeError('API request must remain on the configured origin');
  }
  return url;
}
