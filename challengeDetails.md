
Challenge Overview
Welcome to the Code Completion Context Composition Competition
Announcements

June 5, 2025: Please join our Discord server to instantly read the announcements and reach out to the organising team.

July 17, 2025: All submissions outperforming the "baseline-recent" baseline in the leaderboard will be eligible for a workshop paper.
Overview

This competition focuses on improving code completion by developing better context composition techniques. The goal is to create methods that can effectively compose relevant context from a codebase to help language models generate better code completions.
Why This Matters

Code completion is a crucial developer productivity tool, but its effectiveness heavily depends on the context provided to the language model. By improving context composition, we can make code completion more accurate and useful for real-world development scenarios.
Competition Structure

The competition consists of three phases:

    Practice Phase May 30 — July 25, 2025
        Get familiar with the submission format
        Test your solutions with immediate feedback
        No restrictions on number of submissions
    Public Phase June 9 — July 25, 2025
        Main competition phase
        Public leaderboard
        Limited number of submissions per day
    Private Phase July 25 — August 18, 2025
        Final evaluation phase
        Docker-based submissions
        Most restricted submission limits

Prizes

Each track awards prizes to the top three teams:

    🥇 1st place: USD $3,000
    🥈 2nd place: USD $2,000
    🥉 3rd place: USD $1,000

That’s a USD $12,000 prize pool in total, plus free ASE 2025 workshop registration for a representative from each top team per track.

Winners will also receive:

    🎁 A 1-year JetBrains All Products Pack license for every team member (12 IDEs, 3 extensions, 2 profilers; worth $289 for individual use).
    🔑 API keys for all Mistral AI models on La Plateforme, for you to use however you like.

Getting Started

    Download the starting kit
    Read the README
    Try the example practice phase submission




Evaluation Criteria
Evaluation Details
Evaluation Process

Submissions are evaluated through the following process:

    Your submission is validated for format correctness
    The composed contexts are used with three different models:
        Mellum by JetBrains
        Codestral by Mistral AI
        Qwen2.5-Coder by Alibaba Cloud
    Each model generates completions based on your composed context
    Completions are evaluated using the ChrF metric against the ground truth
    The final score is the average ChrF score across all three models

Handling the submitted contexts

We use FIM (fill in the middle) when prompting models to do code completion. We handle the preparation of a correct prompt for each model on the evaluation side: introducing correct special tokens, arranging context, prefix, and suffix in a correct sequence based on the model, and so on. We trim the user-submitted context from the left to fit the context window of the respective model (8K tokens for Mellum, 16K for Codestral and Qwen).

We ask contestants to use the special file separator token from Qwen2.5 <|file_sep|>. When prompting models, we replace it with the special token used by the specific model.
Metric: ChrF Score

The ChrF score is a character n-gram F-score that measures the similarity between generated and reference text. It is particularly well-suited for code completion evaluation as it captures both precision and recall at the character level, making it sensitive to small but important differences in code syntax.
Leaderboard

The leaderboard displays the following metrics:

    Average ChrF Score: The primary ranking metric (average across all models)
    Mellum ChrF Score: ChrF score for Mellum-4B
    Codestral ChrF Score: ChrF score for Codestral-2501
    Qwen-Coder ChrF Score: ChrF score for Qwen2.5-Coder-7B

Submission Requirements

Your submission must meet these requirements:

    Correct JSONL format
    No malicious code
    Reasonable execution time
    Adherence to the competition rules

