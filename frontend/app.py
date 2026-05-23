"""Streamlit UI — thin client over the FastAPI backend.

Implement:
  - text box for the prompt + a slider for max_new_tokens
  - on submit: requests.post() to the FastAPI /generate endpoint
  - show the returned text

Do NOT load the model here. The frontend only calls the API.
Run: streamlit run frontend/app.py
"""
