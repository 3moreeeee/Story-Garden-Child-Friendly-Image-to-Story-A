# Story Garden Web App

This folder contains the Streamlit front end for the Story Garden project.

It is the interactive layer that turns the original CNN-LSTM captioning notebook into a child-friendly website with story generation and quiz creation.

## Run

```bash
streamlit run webapp/app.py
```

## What it includes

- Caption generation from the saved CNN-LSTM model.
- A moderation slot for future safety logic.
- A child-friendly rewrite stage.
- Story generation.
- Quiz generation.

## Notes

- The UI is intentionally designed to be colorful, simple, and classroom-friendly.
- The pipeline is built to keep working even when optional story or quiz checkpoints are unavailable.
- The main project documentation lives in the top-level [README.md](../README.md).
