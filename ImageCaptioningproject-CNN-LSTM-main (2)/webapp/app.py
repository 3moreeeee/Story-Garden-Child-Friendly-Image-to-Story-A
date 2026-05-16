from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from pipeline import dataset_labels_for, load_sample_images, run_pipeline


st.set_page_config(
    page_title="Story Garden",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Nunito:wght@400;500;600;700;800&display=swap');

        :root {
          --ink: #21313f;
          --muted: #5f7484;
          --card: rgba(255, 255, 255, 0.78);
          --border: rgba(33, 49, 63, 0.09);
          --shadow: 0 24px 60px rgba(33, 49, 63, 0.12);
          --shadow-soft: 0 16px 30px rgba(33, 49, 63, 0.08);
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(255,255,255,0.95), transparent 24%),
            radial-gradient(circle at 82% 16%, rgba(255, 224, 122, 0.42), transparent 18%),
            radial-gradient(circle at 12% 82%, rgba(94, 215, 161, 0.25), transparent 22%),
            radial-gradient(circle at 88% 78%, rgba(87, 183, 255, 0.22), transparent 18%),
            linear-gradient(135deg, #6fd7ff 0%, #8cf0c8 28%, #fff2b3 60%, #ffc8b4 100%);
          color: var(--ink);
        }

        html, body, [class*='css'] {
          font-family: 'Nunito', sans-serif;
          color: var(--ink);
        }

        .block-container {
          padding-top: 1.6rem;
          padding-bottom: 2.2rem;
        }

        [data-testid='stHeader'], [data-testid='stToolbar'] {
          background: transparent;
          visibility: hidden;
          height: 0;
        }

        [data-testid='stSidebar'] {
          background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.80));
          border-right: 1px solid rgba(33, 49, 63, 0.08);
          backdrop-filter: blur(16px);
        }

        [data-testid='stSidebar'] > div:first-child {
          padding-top: 1.2rem;
        }

        .hero {
          position: relative;
          overflow: hidden;
          background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.76));
          border: 1px solid var(--border);
          border-radius: 34px;
          padding: 32px 32px 22px;
          box-shadow: var(--shadow);
          margin-bottom: 20px;
        }

        .hero::before,
        .hero::after {
          content: '';
          position: absolute;
          border-radius: 999px;
          opacity: 0.9;
          pointer-events: none;
        }

        .hero::before {
          width: 180px;
          height: 180px;
          right: -52px;
          top: -56px;
          background: radial-gradient(circle, rgba(255,224,122,0.55), rgba(255,224,122,0.1) 70%, transparent 71%);
        }

        .hero::after {
          width: 120px;
          height: 120px;
          left: -26px;
          bottom: -42px;
          background: radial-gradient(circle, rgba(87,183,255,0.5), rgba(87,183,255,0.08) 70%, transparent 72%);
        }

        .hero-grid {
          display: grid;
          grid-template-columns: 1.55fr 0.85fr;
          gap: 18px;
          align-items: end;
          position: relative;
          z-index: 1;
        }

        .hero-badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: rgba(255,255,255,0.90);
          border: 1px solid rgba(33,49,63,0.08);
          border-radius: 999px;
          padding: 7px 12px;
          font-size: 0.84rem;
          font-weight: 800;
          letter-spacing: 0.02em;
          box-shadow: var(--shadow-soft);
          margin-bottom: 12px;
        }

        .hero h1 {
          font-family: 'Fredoka', sans-serif;
          font-size: 3.35rem;
          line-height: 1.05;
          margin-bottom: 8px;
          letter-spacing: -0.04em;
        }

        .hero p {
          font-size: 1.03rem;
          max-width: 920px;
          margin: 0;
          color: var(--muted);
          line-height: 1.65;
        }

        .pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 18px;
        }

        .pill {
          background: rgba(255,255,255,0.84);
          border: 1px solid var(--border);
          padding: 9px 14px;
          border-radius: 999px;
          font-weight: 700;
          box-shadow: 0 10px 20px rgba(33, 49, 63, 0.05);
        }

        .hero-note {
          background: rgba(33,49,63,0.04);
          border: 1px solid rgba(33,49,63,0.07);
          border-radius: 22px;
          padding: 16px 18px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }

        .hero-note strong {
          display: block;
          font-family: 'Fredoka', sans-serif;
          font-size: 1.1rem;
          margin-bottom: 4px;
        }

        .hero-note span {
          color: var(--muted);
          font-size: 0.95rem;
          line-height: 1.55;
        }

        .stat-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin-top: 18px;
          position: relative;
          z-index: 1;
        }

        .stat-card {
          background: rgba(255,255,255,0.88);
          border: 1px solid var(--border);
          border-radius: 22px;
          padding: 14px 15px;
          box-shadow: var(--shadow-soft);
        }

        .stat-card strong {
          display: block;
          font-family: 'Fredoka', sans-serif;
          font-size: 1.1rem;
          margin-bottom: 3px;
        }

        .stat-card span {
          font-size: 0.88rem;
          color: var(--muted);
        }

        .stage-card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 24px;
          padding: 18px 18px 16px;
          box-shadow: var(--shadow-soft);
          transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
          min-height: 100%;
        }

        .stage-card:hover {
          transform: translateY(-3px);
          box-shadow: var(--shadow);
          border-color: rgba(33,49,63,0.14);
        }

        .stage-card h3 {
          font-family: 'Fredoka', sans-serif;
          margin-top: 0;
          margin-bottom: 8px;
          font-size: 1.18rem;
        }

        .tiny {
          font-size: 0.93rem;
          color: var(--muted);
          line-height: 1.45;
        }

        .step-grid {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 12px;
          margin: 12px 0 16px;
        }

        .step {
          background: rgba(255,255,255,0.82);
          border-radius: 18px;
          padding: 15px 14px;
          text-align: center;
          border: 1px solid rgba(33,49,63,0.08);
          box-shadow: var(--shadow-soft);
        }

        .step strong {
          display: block;
          font-family: 'Fredoka', sans-serif;
          font-size: 1rem;
          margin-bottom: 4px;
        }

        .step span {
          color: var(--muted);
          font-size: 0.88rem;
          line-height: 1.4;
        }

        .section-title {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin: 10px 0 12px;
        }

        .section-title h2 {
          font-family: 'Fredoka', sans-serif;
          font-size: 1.45rem;
          margin: 0;
        }

        .section-title p {
          margin: 0;
          color: var(--muted);
          font-size: 0.92rem;
        }

        .panel {
          background: rgba(255,255,255,0.84);
          border: 1px solid rgba(33,49,63,0.09);
          border-radius: 28px;
          padding: 18px 18px 8px;
          box-shadow: var(--shadow-soft);
        }

        .story-block {
          background: linear-gradient(180deg, rgba(255,255,255,0.93), rgba(255,255,255,0.84));
          border: 1px solid rgba(33,49,63,0.08);
          border-radius: 26px;
          padding: 18px;
          box-shadow: var(--shadow-soft);
        }

        .story-text {
          white-space: pre-wrap;
          line-height: 1.68;
          font-size: 1rem;
          color: var(--ink);
        }

        .split-card {
          background: rgba(255,255,255,0.84);
          border: 1px solid rgba(33,49,63,0.08);
          border-radius: 22px;
          padding: 14px 14px 10px;
          box-shadow: var(--shadow-soft);
          height: 100%;
        }

        .answer-badge {
          display: inline-block;
          background: linear-gradient(135deg, #1b8f58, #2bb673);
          color: white;
          border-radius: 999px;
          padding: 3px 10px;
          margin-left: 8px;
          font-size: 0.78rem;
        }

        .gallery-card {
          background: rgba(255,255,255,0.86);
          border: 1px solid rgba(33,49,63,0.08);
          border-radius: 20px;
          padding: 10px;
          box-shadow: var(--shadow-soft);
          margin-bottom: 10px;
        }

        .footer-note {
          margin-top: 14px;
          color: var(--muted);
          font-size: 0.9rem;
        }

        @media (max-width: 900px) {
          .hero-grid {
            grid-template-columns: 1fr;
          }

          .hero h1 {
            font-size: 2.25rem;
          }

          .stat-row {
            grid-template-columns: 1fr;
          }

          .step-grid {
            grid-template-columns: 1fr 1fr;
          }

          .section-title {
            flex-direction: column;
            align-items: flex-start;
          }
        }

        @media (max-width: 640px) {
          .step-grid {
            grid-template-columns: 1fr;
          }

          .hero {
            padding: 24px 20px 18px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def image_to_pil(uploaded) -> Image.Image:
    return Image.open(BytesIO(uploaded.getvalue())).convert("RGB")


def render_section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class='section-title'>
          <div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-grid">
            <div>
              <div class="hero-badge">Story Garden · image-to-story playground</div>
              <h1>Make pictures feel like a bright little book.</h1>
              <p>
                A child-friendly website that turns an image into a caption, softens the language,
                grows it into a story, and ends with a playful quiz. Everything is arranged as a
                single colorful journey so the pipeline feels easy to explore.
              </p>
              <div class="pill-row">
                <span class="pill">Image captioning</span>
                <span class="pill">Safety pass</span>
                <span class="pill">Kid-friendly rewriting</span>
                <span class="pill">Story generation</span>
                <span class="pill">Quiz maker</span>
              </div>
            </div>
            <div class="hero-note">
              <strong>Designed for kids, teachers, and demos</strong>
              <span>
                The page uses soft glass cards, friendly colors, and a simple flow so the whole
                experience feels cheerful instead of technical.
              </span>
            </div>
          </div>
          <div class="stat-row">
            <div class="stat-card"><strong>5 stages</strong><span>From caption to quiz in one path</span></div>
            <div class="stat-card"><strong>Soft visuals</strong><span>Large cards, gentle spacing, clear focus</span></div>
            <div class="stat-card"><strong>Child first</strong><span>Friendly wording and playful presentation</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_grid() -> None:
    cols = st.columns(5, gap="small")
    items = [
        ("1. Caption", "Describe the picture.", "🖼️"),
        ("2. Safety", "Check if the text is gentle.", "🛡️"),
        ("3. Kid Words", "Rewrite with child-friendly words.", "🌟"),
        ("4. Story", "Turn the caption into a story.", "📖"),
        ("5. Quiz", "Ask fun questions.", "❓"),
    ]
    for col, (title, desc, icon) in zip(cols, items):
        accent = {
            "🖼️": "linear-gradient(135deg, rgba(87,183,255,0.22), rgba(87,183,255,0.08))",
            "🛡️": "linear-gradient(135deg, rgba(94,215,161,0.22), rgba(94,215,161,0.08))",
            "🌟": "linear-gradient(135deg, rgba(255,224,122,0.28), rgba(255,224,122,0.10))",
            "📖": "linear-gradient(135deg, rgba(255,140,102,0.22), rgba(255,140,102,0.08))",
            "❓": "linear-gradient(135deg, rgba(127,121,255,0.18), rgba(127,121,255,0.08))",
        }[icon]
        with col:
            st.markdown(
                f"<div class='stage-card' style='background:{accent};'><h3>{icon} {title}</h3><div class='tiny'>{desc}</div></div>",
                unsafe_allow_html=True,
            )


def render_quiz(quiz_items: list[dict]) -> None:
    if not quiz_items:
        st.info("No quiz questions were parsed.")
        return

    for index, item in enumerate(quiz_items, 1):
        with st.container(border=True):
            st.markdown(f"**Q{index}. {item.get('question', '')}**")
            options = item.get("options", {})
            for letter in ["A", "B", "C", "D"]:
                if letter in options:
                    label = options[letter]
                    if item.get("answer") == letter:
                        st.markdown(
                            f"- {letter}. {label} <span class='answer-badge'>answer</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"- {letter}. {label}", unsafe_allow_html=True)


inject_styles()
render_hero()
render_stage_grid()

with st.sidebar:
    st.markdown("### Choose a picture")
    st.caption("Pick a sample or upload your own image to start the story path.")
    samples = load_sample_images(limit=18)
    sample_names = [str(path.relative_to(path.parents[1])) for path in samples]
    selected_sample = st.selectbox("Sample image", ["Upload your own"] + sample_names)
    uploaded = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "webp"])
    show_model_notes = st.checkbox("Show model notes", value=True)
    generate_quiz_flag = st.checkbox("Generate quiz", value=True)
    show_dataset_hints = st.checkbox("Show dataset hints", value=True)
    st.markdown("---")
    run_button = st.button("Generate the full pipeline", type="primary", use_container_width=True)


selected_image = None
selected_image_name = None
if uploaded is not None:
    selected_image = image_to_pil(uploaded)
    selected_image_name = uploaded.name
elif selected_sample != "Upload your own":
    sample_path = samples[sample_names.index(selected_sample)]
    selected_image = Image.open(sample_path).convert("RGB")
    selected_image_name = sample_path.name


tabs = st.tabs(["Pipeline", "Gallery", "About"])

with tabs[0]:
    render_section_title("Pipeline", "A polished walkthrough of the caption, safety, story, and quiz stages.")
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Preview")
        if selected_image is not None:
            st.image(selected_image, use_container_width=True)
            st.caption(selected_image_name or "Uploaded image")
        else:
            st.info("Choose a sample or upload an image to begin.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Pipeline output")
        if run_button and selected_image is not None:
            with st.spinner("Painting the story pipeline..."):
                result = run_pipeline(selected_image, image_name=selected_image_name)
            st.session_state["latest_result"] = result
        else:
            result = st.session_state.get("latest_result")

        if result is not None:
            st.markdown("#### 1. Caption")
            st.markdown(f"<div class='story-block'><div class='story-text'>{result.caption}</div></div>", unsafe_allow_html=True)

            st.markdown("#### 2. Safety")
            st.write(f"Status: **{result.moderation_status}**")
            st.write(f"Score: {result.moderation_score:.2f}")
            if result.notes:
                for note in result.notes:
                    st.caption(f"• {note}")

            st.markdown("#### 3. Kid-friendly version")
            before_col, after_col = st.columns(2)
            with before_col:
                st.markdown("<div class='split-card'>", unsafe_allow_html=True)
                st.caption("Original")
                st.write(result.caption)
                st.markdown("</div>", unsafe_allow_html=True)
            with after_col:
                st.markdown("<div class='split-card'>", unsafe_allow_html=True)
                st.caption("Child-friendly")
                st.write(result.child_caption)
                st.markdown("</div>", unsafe_allow_html=True)
            if result.child_caption == result.caption:
                st.info("This caption was already gentle, so the child-friendly step kept the wording nearly the same.")
            else:
                st.success("The caption was softened for children.")

            st.markdown("#### 4. Story")
            st.markdown(f"<div class='story-block'><div class='story-text'>{result.story}</div></div>", unsafe_allow_html=True)

            st.markdown("#### 5. Quiz")
            if generate_quiz_flag:
                render_quiz(result.quiz_items)
            else:
                st.info("Quiz generation is turned off in the sidebar.")

            st.download_button(
                "Download story",
                data=result.story,
                file_name="children_story.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.download_button(
                "Download quiz",
                data=result.quiz_text,
                file_name="children_quiz.txt",
                mime="text/plain",
                use_container_width=True,
            )

            if show_model_notes:
                with st.expander("What the app is using behind the scenes", expanded=False):
                    st.markdown(
                        "- Captioning model: CNN-LSTM restored from the saved `best_model.h5` weights.\n"
                        "- Story model: TinyLlama LoRA adapter when a compatible GPU is available; otherwise a child-safe fallback story generator.\n"
                        "- Quiz model: Flan-T5 quiz generator when the local checkpoint loads successfully; otherwise a fallback quiz template.\n"
                        "- Safety and vocabulary stage: rule-based bridge ready for your lab module, so the website already has the pipeline slot in place."
                    )
        else:
            st.info("Run the pipeline to see the caption, safety pass, story, and quiz.")
        st.markdown("</div>", unsafe_allow_html=True)

    if show_dataset_hints and selected_sample != "Upload your own" and selected_image_name:
        hints = dataset_labels_for(selected_image_name)
        if hints:
            render_section_title("Dataset hints", "A quick look at labels that feel close to the selected sample.")
            cols = st.columns(min(3, len(hints)), gap="small")
            for col, (label, score) in zip(cols, hints[:3]):
                with col:
                    st.markdown(
                        f"<div class='stage-card'><h3>{label.title()}</h3><div class='tiny'>Similarity: {float(score):.2f}</div></div>",
                        unsafe_allow_html=True,
                    )

with tabs[1]:
    render_section_title("Sample gallery", "A few images from the workspace so you can test the pipeline quickly.")
    gallery = load_sample_images(limit=12)
    if gallery:
        rows = [gallery[i : i + 4] for i in range(0, len(gallery), 4)]
        for row in rows:
            cols = st.columns(len(row), gap="small")
            for col, image_path in zip(cols, row):
                with col:
                    st.markdown("<div class='gallery-card'>", unsafe_allow_html=True)
                    image = Image.open(image_path).convert("RGB")
                    st.image(image, use_container_width=True)
                    st.caption(image_path.name)
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No sample images were found in the workspace.")

with tabs[2]:
    render_section_title("About", "What is included in the Story Garden pipeline.")
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        """
        This website is organized around your pipeline:

        1. Image captioning from the saved CNN-LSTM model.
        2. A moderation stage that can be replaced by your content-safety lab.
        3. A vocabulary softening stage for child-friendly phrasing.
        4. Children’s story generation with the LoRA story adapter.
        5. Quiz generation with the Flan-T5 quiz module.

        The app is designed so you can plug in your remaining lab files later without redesigning the interface.
        """
    )
    st.code(
        "Image -> Caption -> Safety -> Child Vocabulary -> Story -> Quiz",
        language="text",
    )
    st.markdown("<div class='footer-note'>Built as a friendly, colorful front end for the image-to-story workflow.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)