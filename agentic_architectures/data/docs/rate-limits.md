# Rate limits and retries

APIs answer HTTP 429 when a client sends too many requests. Retry with exponential
backoff and jitter, and honour the `Retry-After` header when it is present. Only retry
requests that are safe to repeat: reads, or writes protected by an idempotency key.
