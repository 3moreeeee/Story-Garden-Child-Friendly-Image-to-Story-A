# Story Garden

Story Garden is a child-friendly image-to-story web app. It takes an uploaded image, generates a caption, gently rewrites anything that feels too advanced, turns it into a short story, and finishes with a simple quiz.

## What Is Included

- `webapp/app.py`: the Streamlit interface.
- `webapp/pipeline.py`: captioning, moderation, story generation, and quiz generation.
- `webapp/requirements.txt`: Python dependencies for the app.
- `data/captions.txt`: caption lookup data.
- `Images/` and `Images_food/`: sample images used by the app.
- `ImageCaptioningproject-CNN-LSTM-main/best_model.h5`: CNN-LSTM captioning weights.
- `ImageCaptioningproject-CNN-LSTM-main/vocab.pkl`: caption vocabulary.
- `ImageCaptioningproject-CNN-LSTM-main/quiz/`: local quiz model files.
- `story_lora_adapter/`: local story-generation adapter files.
- Notebooks in the repository are kept for experimentation and reference.

## How It Works

1. Upload an image or choose one of the sample images.
2. The captioning model generates a caption.
3. The moderation pass checks whether the text needs a softer rewrite.
4. The child-friendly rewrite turns the caption into a gentle prompt.
5. The story generator creates a short children’s story.
6. The quiz generator returns a simple five-question quiz.

## Run Locally

1. Install Python 3.10 or newer.
2. Install dependencies:

	```bash
	pip install -r webapp/requirements.txt
	```

3. Start the app from the project root:

	```bash
	streamlit run webapp/app.py
	```

## Notes

- The app uses local model files when they are available.
- If the story or quiz model cannot load, the app falls back to safe rule-based output so it still runs.
- Notebooks are included for the original training and experimentation workflow.

## Project Goal

This repo keeps the launch surface small: the Streamlit app, the notebooks, and the model/data files needed to run Story Garden end to end.
