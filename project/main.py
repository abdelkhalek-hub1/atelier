#!/usr/bin/env python3
"""
Main entry point to execute the LangGraph Wikipedia Workflow from the terminal.
Usage:
    python main.py "your question here"
"""

import sys
from app.graph import run_workflow


def main() -> None:
    # If a question was provided as a command-line argument, use it.
    # Otherwise, prompt the user for input.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("💡 TIP: You can also pass the question directly: python main.py 'your question'")
        question = input("❓ Enter your question: ").strip()

    if not question:
        print("❌ Error: Question cannot be empty.")
        sys.exit(1)

    print(f"\n🚀 Running workflow for: '{question}'...")
    print("-" * 60)

    try:
        # Run the workflow
        state = run_workflow(question)

        # Output the result
        print("\n📊 WORKFLOW RESULT:")
        print(f"🆔 Execution ID : {state.execution_id}")
        print(f"📅 Timestamp    : {state.timestamp}")

        if state.error:
            print(f"❌ Status       : FAILED")
            print(f"⚠️ Error        : {state.error}")
            print(f"📝 Final Answer : {state.final_answer}")
        else:
            print(f"✅ Status       : SUCCESS")
            print(f"📝 Final Answer :\n\n{state.final_answer}")

    except Exception as exc:
        print(f"\n❌ Workflow execution failed with critical exception: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
