# Idempotency keys

An idempotency key is a unique value a client sends with a request that has side
effects, such as creating a payment. If the request times out, the client retries with
the same key; the server recognises the key, does not charge again, and returns the
stored result of the first request. Keys usually expire after 24 hours.
