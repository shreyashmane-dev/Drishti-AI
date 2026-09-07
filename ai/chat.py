"""
DRISHTI AI - Live Interactive Retinal Analysis & Conversational AI Chat CLI.

Usage:
  python ai/chat.py --image data/aptos2019/real_images/002c21358ce6.png
  python ai/chat.py
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import DATA_DIR
from ai.conversational_agent import DrishtiChatSession


def select_default_image() -> Path:
    """Finds an available real clinical fundus image."""
    real_images_dir = DATA_DIR / "aptos2019" / "real_images"
    if real_images_dir.exists():
        images = list(real_images_dir.glob("*.png")) + list(real_images_dir.glob("*.jpg"))
        if images:
            return images[0]
    sample_images_dir = DATA_DIR / "aptos2019" / "train_images"
    if sample_images_dir.exists():
        images = list(sample_images_dir.glob("*.png"))
        if images:
            return images[0]
    raise FileNotFoundError("No sample retinal fundus image found in data/aptos2019. Please pass --image.")


def main():
    parser = argparse.ArgumentParser(description="DRISHTI AI - Interactive ChatGPT-Style Vision & Clinical Chat")
    parser.add_argument("--image", type=str, default=None, help="Path to retinal fundus image (e.g. 002c21358ce6.png)")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else select_default_image()
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return

    print("=" * 70)
    print("👁️  DRISHTI AI - Generative Vision & Clinical Dialogue System")
    print("=" * 70)
    print(f"📁 Selected Fundus Scan: {image_path}")

    session = DrishtiChatSession(image_path)
    pred = session.analysis_result.get("clinical_prediction", {})
    triage = session.analysis_result.get("referral_triage", {})

    print("\n" + "=" * 70)
    print(f"🔍 SCAN DIAGNOSIS: {pred.get('icdr_label')} (Grade {pred.get('icdr_grade')})")
    print(f"📊 Confidence: {pred.get('confidence', 0)*100:.1f}% | Risk Urgency: {triage.get('urgency')}")
    print("=" * 70)
    print("\n🤖 DrishtiAI Initial Synthesis:")
    print(session.initial_report)
    print("\n" + "-" * 70)
    print("💬 You can now chat with DrishtiAI about your scan!")
    print("   Type your question, or type '/image <path>' to analyze a new scan.")
    print("   Type 'exit' or 'quit' to end the session.")
    print("-" * 70 + "\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                print("\n👋 DrishtiAI: Stay healthy and remember to schedule your regular eye screenings!")
                break

            if user_input.startswith("/image "):
                new_path = Path(user_input.split(" ", 1)[1].strip())
                if not new_path.exists():
                    print(f"❌ File not found: {new_path}")
                    continue
                print(f"\n🔄 Switching to new fundus image: {new_path.name}...")
                session = DrishtiChatSession(new_path)
                print("\n🤖 DrishtiAI:")
                print(session.initial_report)
                continue

            # Generate dynamic AI response
            print("\n🤖 DrishtiAI is typing...\n")
            response = session.chat(user_input)
            print(f"🤖 DrishtiAI:\n{response}")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting DrishtiAI Chat.")
            break


if __name__ == "__main__":
    main()
