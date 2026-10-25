# Agent budgets

An agent loop should stop on its own terms, not the model's. Give every loop a maximum
number of LLM calls, tool calls, tokens and seconds, and stop when the same action
repeats without progress. Return the best answer so far with the reason it stopped.
