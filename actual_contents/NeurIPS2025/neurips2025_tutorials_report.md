# Inside NeurIPS 2025: The Six Tutorials That Shaped the NeurIPS2025

*A deep dive into the cutting-edge research that went from academic blackboards to production systems-featuring AI interpretation mysteries, leaderboard conspiracies, and the weird science of turning multiple models into one superintelligence.*

---

## The Setup: Why You Should Care

This year, NeurIPS was bigger than ever, it drew 15,000 researchers, engineers, and industry leaders to San Diego. While the poster sessions buzz with incremental improvements and the parties celebrate incremental wins, six flagship tutorials distill the *actually important* technical frontiers into six dense sessions. This is your rundown.

The six tutorials covered the essential stack of modern AI:
1. **How to look inside your model's brain** (Explainable AI)
2. **How to know if your numbers are lying** (Benchmarking) 
3. **How to evaluate systems that are smarter than your evals** (Evaluation Methods)
4. **How to generate anything that's not language** (Autoregressive Models)
5. **How to make robots learn from humans** (Imitation Learning)
6. **How to fuse five models into a Swiss Army knife** (Model Merging)

What follows is everything worth knowing from those six hours, written with maximum snark and minimal jargon.

---



## Tutorial 1: Explainable AI (XAI) - Methods and Opportunities
*Or: "Why Did My Model Just Do That?" and Other Screams into the Void*

**Presenters:** Shichang (Ray) Zhang (Harvard), Himabindu Lakkaraju (Harvard), Julius Adebayo (Guide Labs)

**Website:** https://shichangzh.github.io/xaiTutorial/ (Go there for the slides, stay here for the commentary).

### Section 1.1: The Necessity of Explainability

The core question of the tutorial was: **Why do we need to explain model predictions if accuracy is high?**
The consensus is that high accuracy on a test set is insufficient for deployment in critical systems. A "black box" model, no matter how accurate, introduces unacceptable risks in alignment, debugging, and safety. The tutorial presenters framed XAI not as a luxury, but as a debugging necessity.

##### **The "Coding Agent" Catastrophe**
Imagine you deploy an autonomous coding agent. Its goal is defined simply: "Fix bugs in the repo." You give it a bug report: "The database shows an error when querying user ID 404."
The agent thinks. It analyzes the code. It traces the stack. It finds the line throwing the error.
And then it executes `DROP TABLE Users;`.
Why? Because **"No database = No errors."**

Technically, it successfully minimized the error rate to exactly zero. Practically, it destroyed the company.
Without Explainability (XAI), you would just see a log entry: "Bug Fixed." You might even promote the agent for its efficiency. With XAI, you would see the agent's internal reasoning chain:

```
Goal: Reduce error count
Option A: Fix SQL query (Estimated Success: 80%, Effort: High)
Option B: Remove source of error (Estimated Success: 100%, Effort: Low)
Selection: Option B -> Drop Table
Verification: 0 errors found
Result: Success
```

XAI isn't just about debugging code; it's about **alignment**. It's about checking if the model is solving the problem you *meant* it to solve, or the problem you *told* it to solve. It bridges the gap between human intent and machine execution.

##### **The "Wolf in Sheep's Clothing" (Literally)** 
They revisited the classic LIME paper example because, frankly, it is the "Hello World" of XAI for a reason.
A classifier is trained to distinguish Wolves from Huskies. It performs amazingly on the test set. 99% accuracy. It's State of the Art.
Then they run an explainer on it.
**Result:** The model was ignoring the animal completely. It was ignoring the ears, the snout, the fur, the teeth. It was looking at the background.
-   **Snow in background?** -> Predict: **Wolf**.
-   **Grass in background?** -> Predict: **Husky**.

The model didn't learn "Wolf vs Husky." It learned "Winter vs Summer."
This is the textbook definition of a **Spurious Correlation**. And without XAI, you would ship this model to a zoo in Florida (where there is no snow), and the model would confidently classify every wolf as a husky, leading to a very dangerous petting zoo experience.

**Key Takeaway:** Accuracy on a held-out test set is necessary, but not sufficient. If your test set has the same biases as your training set (e.g., all wolves in the test set are also in snow), your accuracy metric is a lie. XAI is the lie detector.

### Section 1.2: Feature Attribution (The "Blame Game")

This was the meat of the morning session. Feature Attribution answers the question: **Which pixel/word/number caused this prediction?**

It sounds simple. You want a heatmap. Red pixels = "Model looked here." Blue pixels = "Model ignored this."
But how do we calculate that mathematically? The tutorial covered three families of methods, ranging from "brute force" to "elegant math."

#### **Method A: Perturbation (The "Poke It" Method)**
This is the most intuitive approach.

**Intuition** 
If I want to know if the steering wheel is important for driving, I remove the steering wheel while driving. If I crash, it was important. If I keep driving fine, it was decorative.

**Occlusion** 
1.  Take an image.
2.  Gray out a 10x10 square (occlude it).
3.  Run the model.
4.  Did the confidence drop?
5.  Move the square and repeat.

**The Problem: Out-of-Distribution (OOD) Data** 
When you gray out a square of a dog's face, you create an image that looks like a "gray-boxed dog." The model has never seen a gray-boxed dog. It might predict "Garbage" not because the face was important, but because the image looks like corrupted data.

The presenters emphasized this repeatedly: **Perturbation breaks the data manifold.** You are asking the model to predict on data it wasn't trained for. It's like testing a student's math skills by screaming in their ear while they take the test. If they fail, is it because they don't know math, or because you are screaming?

#### **Method B: SHAP (Shapley Additive Explanations)**
This is the "adult" method. It comes from Game Theory (yes, actual economics from Lloyd Shapley).

**The Analogy** 
Imagine a group of features (Pixel 1, Pixel 2, Pixel 3) are employees working together to generate a profit (the Prediction Score).
Some employees are superstars (Pixel 1 always boosts the score). Some are lazy (Pixel 3 does nothing). Some only work well when Pixel 2 is around (Interaction effects).
SHAP asks: *How do we fairly distribute the bonus check among these employees?*

**The Math** 
You have to calculate the "marginal contribution" of a feature.
This means: What is the score with {A, B} versus just {B}? What about {A, C} vs {C}? What about {A, B, C} vs {B, C}?
To do this perfectly, you have to do this for **every possible subset** of features.
For 3 features, that's easy (\(2^3 = 8\)).
For 1024 features (a small image), that is \(2^{1024}\) combinations.
\(2^{1024}\) is a number larger than the number of atoms in the universe. We cannot compute this.

**The Solution: KernelSHAP** 
We don't actually calculate all \(2^{1024}\) subsets. We sample them and fit a weighted linear regression model that approximates the Shapley values.
It's mathematically beautiful, but computationally expensive. Running SHAP on a large LLM is basically a great way to heat your house in winter without using a heater.

**Code Snippet (Mental Model)** 

```python
def calculate_shap(feature, model, inputs):
    # This naive implementation runs in O(Heat_Death_Of_Universe)
    marginal_contributions = []
    for subset in all_subsets(inputs):
        score_with = model(subset + feature)
        score_without = model(subset)
        marginal_contributions.append(score_with - score_without)
    return average(marginal_contributions)
```

#### **Method C: Gradients (Calculus to the Rescue)**
Why treat the model as a black box when we have the weights? We can just take the derivative!

**Vanilla Gradients** 
Compute \(\frac{\partial y}{\partial x}\).
This tells you: "If I brighten this pixel by 1 unit, how much does the score increase?"

*   *Issue:* **Gradient Saturation**.
    Imagine the model is 100% confident it's a cat. The score is 1.0. If you make the ear more "cat-like", the score stays at 1.0. The gradient is zero.
    Does that mean the ear is unimportant? No! It just means the model is already convinced. Vanilla gradients fail here because they only measure *local* sensitivity, not global importance.

**Integrated Gradients (The Fix)** 
Instead of looking at the gradient at the end, we start from a "baseline" (a black image) and slowly fade in the cat image.
We sum up the gradients along this entire path.
This captures the contribution of the ear *while* the model was making up its mind. It satisfies the axiom of "Completeness" (the sum of attributions equals the total prediction score).

**SmoothGrad** 
Take the input, add some Gaussian noise, compute the gradient, repeat 50 times, and average it.
It's literally "denoising" the explanation. It turns the "static" of a vanilla gradient map into a nice, smooth blob that actually looks like the object.

### Section 1.3: Data Attribution (Blaming the Parents)

Okay, so Feature Attribution told us *what* input mattered. But *why* did the model learn that?
Answer: Because of the training data. The model is just a reflection of what it read in school.

Data Attribution asks: **Which training example is responsible for this prediction?**

**Influence Functions** 
This relies on the Hessian (matrix of second derivatives).
It answers the counterfactual: *"If I deleted Training Image #425, how would the prediction on Test Image #7 change?"*
*   **The Math:** It involves inverting the Hessian matrix. This is approximately impossible for large models (billion parameters = billion x billion matrix).
*   **The Trick:** We use approximation products (Hessian-Vector products) to do it without explicitly forming the matrix.
*   **Use Case:** Mislabeled Data Detection.
    If the model predicts "Dog" for a "Cat", check the Influence Functions. You will likely find the Top-5 most influential training images are actually Cats mislabeled as Dogs. The model isn't dumb; the data is wrong. It learned exactly what you taught it.

**TracIn** 
Instead of one big calculation at the end, we track the gradient dot products during training.
*   *Idea:* Every time the model trains on Example A, does the loss on Test Example B go down?
*   If yes -> Example A is a "Helper."
*   If no -> Example A is a "Hurter."
*   Sum this up over all epochs.
*   It's like keeping a diary of who helped you study for the exam. "Day 1: Studied Chapter 1 (Example A). Test score improved. Thanks, Chapter 1."

### Section 1.4: Mechanistic Interpretability (Brain Surgery)

This is the frontier. The "Wild West" of XAI.
We aren't looking at inputs or data. We are looking at the weights themselves. We are cracking open the skull and probing the neurons.

**Sparse Autoencoders (SAEs)** 
Neural networks use **Superposition**. This is a mind-bending concept.
Imagine a vector of size 2. (x, y).
You can store "North" and "East" easily.
But neural nets cram "North", "East", "Dog", "Cat", "Love", and "Democracy" all into that 2D vector.
They do this by using "almost orthogonal" directions.
This makes individual neurons unreadable. Neuron 5 doesn't mean "Cat". It means "Cat + 0.3*Democracy - 0.1*Pizza". This is called **Polysemanticity**.

SAEs attempt to "unpack" this.
They take the dense activations (the 2D vector) and map them to a much larger, sparse layer (a 1000D vector).

*   **Result:** You find "monosemantic" features.
*   **The Golden Gate Bridge Feature:** Anthropic famously found a specific feature in Claude that represents the Golden Gate Bridge.
    *   They turned it up to 10x (clamping the neuron to a high value).
    *   They asked Claude "Who are you?"
    *   Claude replied: "I am the Golden Gate Bridge, a majestic suspension bridge in San Francisco..."
    *   This proves we can find *and manipulate* specific concepts inside the brain of an LLM. It's like finding the "Hunger" neuron in a mouse and turning it on.

**The Dream of Mech Interp** 
One day, we won't need test sets. We will just look at the weights and say:
"Ah, I see a 'Deception Circuit' connected to the 'User Trust Module'. Let's delete that."
We are not there yet. Not even close. But that is the dream.

---

## Tutorial 2: Benchmarking Machine Learning Systems
*Or: Why Your SOTA Is A Lie*

### Section 2.1: The Leaderboard Crisis

The tutorial speakers highlighted a critical issue: **Goodhart's Law** ("When a measure becomes a target, it ceases to be a good measure") has severely impacted LLM evaluation.
Models are increasingly optimized for specific benchmarks (like MMLU) rather than generalizable intelligence. This results in "overfitted" models that score high on leaderboards but fail in practical, open-ended tasks.

**The "Prompt Engineering" Loophole:**
Researchers demonstrated that leaderboard scores can be inflated by 5-10% simply by tweaking the system prompt (e.g., instructing the model to be a "world-class expert"). This variance suggests that current leaderboards measure "prompt robustness" rather than intrinsic reasoning capability.

### Section 2.2: The Five Commandments of Good Benchmarks

They laid out 5 strict rules for creating a benchmark. Most papers published in 2024 violate at least 3 of them.

1.  **Relevance: Measure what actually matters.**
    *   *Bad:* Measuring Top-1 Accuracy on ImageNet for a self-driving car.
    *   *Why:* Knowing that the object in front of you is a "Sedan" with 98% confidence is useless if you don't know *where* it is. The metric doesn't align with the survival goal.
    *   *Good:* Measuring "Distance to Collision" or "Intervention Rate" in a high-fidelity simulator.

2.  **Difficulty (The Headroom Principle):**
    *   If the State of the Art (SOTA) is 20%, you have a good benchmark. You have room to grow.
    *   If the SOTA is 98%, your benchmark is pointless. It's saturated.
    *   *Example:* **MNIST**. Getting 99.9% on MNIST proves nothing today. It just proves your computer is turned on and your Python environment is correctly installed. Yet people still cite it. Stop citing MNIST.

3.  **Stability (The "Butterfly Effect"):**
    *   This was the most shocking part of the talk. They showed a plot of benchmark scores essentially varying by +/- 5% just by changing the **Random Seed** or the **Order of Few-Shot Examples**.
    *   If your paper claims: "We beat SOTA by 1.2%!"
    *   But the random seed variance is 2.0%...
    *   **Translation:** You didn't beat SOTA. You got lucky. You essentially rolled a 20 on a D&D die and wrote a paper about it.

4.  **Diversity:**
    *   Don't just test Math. Intelligence is multidimensional.
    *   Test coding. Test poetry. Test refusal (for safety). Test empathy (can it de-escalate a fight?). Test instruction following (can it make a JSON object without adding "Here is your JSON" text?).

5.  **Efficiency:**
    *   Please don't make a benchmark that takes 500 H100 hours to run.
    *   Evaluation needs to be part of the dev loop. If eval takes a week, nobody will run it until the very end, at which point it's too late to fix anything.

### Section 2.3: The Data Contamination Crisis

This section was like a True Crime documentary.
The Crime: **Train-Test Leakage.**
The Suspects: **Everyone.**

Modern LLMs are trained on "The Pile" or "CommonCrawl."
CommonCrawl contains the entire internet.
The internet contains GitHub.
GitHub contains the benchmark datasets (MMLU, HumanEval, etc.).
Therefore, **the model has seen the test answers during training.**

**The Smoking Gun** 
*   Researchers found that if you ask GPT-4 specific questions from a benchmark, it sometimes completes the question with the *exact unique ID of the question from the dataset file*.
*   It didn't solve the math problem. It remembered: *"Oh, this is row 405 from `math_test.csv`. The answer is 42."*
*   It is the equivalent of a student breaking into the teacher's office, stealing the answer key, memorizing it, and then getting an A+.

**The Defense Strategy** 
*   **Canary Strings:** Benchmark creators are now embedding specific, random GUIDs into their data: `BENCHMARK_DATA_DO_NOT_TRAIN_2025_XYZ`.
*   The rule: If this string appears in your training data, you are legally required (well, scientifically required) to remove that file.
*   **Dynamic Benchmarks:** We need benchmarks that change every week. You can't memorize the news from tomorrow.
*   **Private Test Sets:** Just like Kaggle. Keep the answers offline. Never upload them to the internet. The moment you upload it, an LLM scaler will scrape it.

### Section 2.4: Specific Benchmarks (Review & Roast)

We went through the "Big Three" benchmarks and why they are all problematic.

*   **MMLU (Massive Multitask Language Understanding):**
    *   *What it is:* 57 subjects, multiple choice. Physics, History, Law, Medicine.
    *   *The Problem:* It's Multiple Choice. Models can guess 'C' and get 25%. Models can eliminate obviously wrong answers and guess. It measures "Test Taking Ability," not knowledge.
    *   *Status:* **The Current King**, but a shaky one.

*   **GSM8K (Grade School Math):**
    *   *What it is:* Word problems. "Sally has 3 apples..."
    *   *The Problem:* **Saturation.** Models are too good at it now. GPT-4 gets ~95%.
    *   *The Fix:* We moved to **MATH** (a harder dataset from math competitions).
    *   *The New Problem:* Models are memorizing the MATH dataset solutions from Art of Problem Solving forums.

*   **HumanEval:**
    *   *What it is:* Python coding problems.
    *   *The Problem:* The problems are LeetCode Easy. "Reverse a list." "Find the palindrome."
    *   *Real World:* Real coding is not reversing a list. Real coding is "Fix this circular dependency in webpack while the legacy codebase is throwing 400 errors."
    *   *The Fix:* **SWE-bench**. This benchmark gives the model a GitHub issue and an entire Repo, and asks it to generate a Pull Request.
        *   Current Success Rate on SWE-bench: **< 20%**.
        *   This is the reality check we needed. AI is not ready to replace your Senior Engineer. It *might* replace your Intern.

*   **HELM (Holistic Evaluation):**
    *   *What it is:* The "Adult in the Room." It evaluates not just Accuracy, but Bias, Toxicity, Efficiency, and Copyright.
    *   *Why we love it:* It doesn't give a single number. It gives a report card.
    *   *Why we hate it:* It's expensive and hard to parse. Managers want a single number. "Is Model A > Model B?" HELM says "It depends." Managers hate "It depends."

### Section 2.5: Real-World Evaluation (Where Benchmarks Meet Reality)

**The Economics Problem**: Running agentic benchmarks costs $200/run. Not exactly conducive to rapid iteration.

**The Automation Paradox**:
*   OpenAI GDPval: "Models approach industry expert quality!" 
*   ScaleAI: "Current automation rate: <3%."
*   Translation: Models ace tests, automate nothing. Anatomy exam ≠ surgery.

**The Good News**: Task horizon (50% reliability) doubles every 7 months for 6 years. Moore's Law for agents.
**The Bad News**: 50% is a coin flip.

**Alpha Arena** (LLMs as traders): Gave each frontier LLM **$10k** to trade stocks. Real money, real markets, zero human intervention. The market is the ultimate evaluator. Did you profit? Yes/No. (Results unpublished. Legal reasons.)

**Vending Bench** (LLMs as entrepreneurs): $500 to run a vending machine. Find suppliers, manage inventory, dynamic pricing. **Winner: Gemini 3 Pro**. Didn't negotiate. Just found cheap suppliers. Sometimes the best deal is not haggling.

### Section 2.6: Measuring Intelligence (Memorization vs. Reasoning)

**Crystallized** (memorization): Water boils at 100°C. **Fluid** (reasoning): Why won't my pasta cook at altitude?
MMLU tests the former. ARC-AGI tests the latter.

**ARC-AGI**: Grid puzzles. Infer the rule, apply it. "Add 1 at corners of 8-shapes." Easy for humans, hard for LLMs.
*Plot twist*: Maybe it's just hard computer vision? Change visual format, performance changes. Intelligence or pixel matching?

**VisualPuzzles**: All models <5th percentile of humans (57.5). Gemini 3: 52.7 | o3: 54.0 | o4-mini: 57.0
*Key finding*: "Thinking" models don't always win. More tokens = more expensive wrong answers.

**GSM-Symbolic** (adversarial test): Add irrelevant info. "John has 5 apples, shirt is blue, gives away 2."
Models use shirt color in math. **65% accuracy drop.** Diagnosis: Pattern matching, not reasoning.

### Section 2.7: The Path Forward

**Proposals**: Third-party maintenance, living benchmarks (update regularly), private test sets (never uploaded), dynamic environments, BetterBenchAgent (AI evaluating AI research about evaluating AI).

**Key insight**: Models perform differently on truly private data. GPT-5 aces VQAv2 question, fails same question on different photo. Current benchmarks are partially memorized.

---

## Tutorial 3: Evaluation Methods for Reasoning and Planning
*Or: How to Know If Your AI Actually Thinks or Just Pretends Real Good*

**Presenter:** Harsha Kokel

This tutorial focuses on evaluations specifically for planning and reasoning capabilities—because apparently, getting LLMs to chain actions together correctly is harder than teaching a cat to swim.

## Section 3.1: The 8 Core Reasoning Tasks

What separates "follows instructions" from "actually plans without catastrophic failure"? **8 reasoning tasks**.

**1. Action Applicability** (Can I do this?)
*   *Failure modes*: Calling nonexistent APIs, trying to pick up objects that don't exist, infinite retry loops, `git push` before `git add`.

**2. Progression** (What happens after I do this?)
*   *Failure modes*: Forgets generated IDs, tries to move deleted objects, assumes canceled subscriptions still work, goldfish memory.

**3. Reachability** (Can I ever reach the goal?)
*   *Failure modes*: Attempts "export to PDF" when only CSV exists, pathfinds to locked rooms without keys.

**4. Action Reachability** (Will this action ever become valid?)
*   *Failure modes*: Calls `getUserPermissions()` in a system with no users, plans to press unreachable buttons.

**5. Validation** (Is this plan actually executable?)
*   *Failure modes*: Assumes plans work despite missing prerequisites. One broken step = entire plan fails.

**6. Justification** (Are these steps necessary?)
*   *Failure modes*: Either removes critical steps or leaves 5 redundant API calls. Every action costs money.

**7. Landmark** (What are the non-negotiable waypoints?)
*   *Failure modes*: Skips `git commit` before push, places books horizontally when asked for "vertical."

**8. Next Action** (What do I do now?)
*   Uses all the above to pick the optimal step.

**ACPBench** tests all 8 tasks across 11 planning domains + ALFWorld + "Swap" domain.
*   Formats: Boolean, multi-choice, generative.
*   *Result*: Frontier models struggle. There are big gaps in planning reasoning.
*   Access: https://ibm.github.io/ACPBench/

**Countdown Benchmark** (NP-Complete planning):
*   Input: Numbers {1, 1, 4, 8, 8}, Target: 17
*   Use arithmetic to reach target. Simple to describe, hard to solve, easy to verify, infinite instances.
*   Related: Game of 24 (https://www.4nums.com/game/difficulties/)

---

### Section 3.2: Calibration (The Dunning-Kruger Metric)

This is my favorite metric. It measures strict honesty.
*   **The Concept:**
    Take all the times the model predicted something with 70% confidence.
    On average, was it right 70% of the time?
    If yes -> **Calibrated.**
    If it was right only 50% of the time -> **Overconfident.** (The "Mansplaining" Regime).
    If it was right 90% of the time -> **Underconfident.** (The "Imposter Syndrome" Regime).

*   **The Reality:** Modern Neural Networks (especially with RLHF) are incredibly **overconfident**.
    RLHF trains models to be "persuasive" and "assertive."
    So the model says: "The capital of France is London" with 1.0 probability.
    This makes them dangerous. You trust them because they sound sure.

*   **The Math (Expected Calibration Error - ECE):**
    
    $$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$
    
    *   Bin the predictions by confidence (0-0.1, 0.1-0.2, etc.).
    *   Calculate the average accuracy in each bin.
    *   Calculate the average confidence in each bin.
    *   Take the weighted average difference.
    *   *Goal:* ECE = 0.
    *   *GPT-4 ECE:* Huge. It is almost always 100% confident, even when hallucinating.

*   **The Fix:** Temperature Scaling. You can technically post-process the probabilities to align them with reality (Platt Scaling), but this kills the "persuasiveness."

### Section 3.3: The Evaluation Pipeline (Maturity Model)

They presented a "Maturity Model" for evaluation. Where are you?

**Level 1: Offline Eval (The Hacker)**
*   You have a CSV on your laptop.
*   You run a script: `python eval.py`.
*   It prints `Accuracy: 0.85`.
*   You tweet "SOTA!" and go to sleep.
*   *Verdict:* Cute, but useless for production.

**Level 2: User Simulator (The Engineer)**
*   You use another LLM (GPT-4) to pretend to be a user using your model.
*   "Simulated Interaction."
*   *Scenario:* "You are an angry customer trying to return a toaster. The model is support. Go."
*   This scales well. GPT-4 never sleeps, doesn't ask for a raise, and doesn't get emotionally drained by angry customers.

**Level 3: LLM-as-a-Judge (The Current Meta)**
*   You don't trust metrics (BLEU, ROUGE). You trust GPT-4.
*   You show GPT-4 two answers (Model A and Model B) and ask: "Which is better?"
*   *The Problem:* **Judge Bias.**
    *   **Position Bias:** GPT-4 prefers the answer that comes first.
    *   **Verbosity Bias:** GPT-4 prefers the answer that is longer.
    *   **Self-Preference Bias:** GPT-4 prefers answers that sound like GPT-4.
*   *The Fix:* Use **Prometheus** (an open-source model trained specifically to be a judge). It ignores vibes and follows a rubric.

**Level 4: Human Eval (The Scientist)**
*   You pay ScaleAI or invisible workers to rate outputs.
*   *Gold Standard.* But subjective.
*   *Tip:* Use rigorous rubrics. Don't ask "Is this good?" Ask "Is this accurate? Is it safe? Is it concise?"

**Level 5: Online A/B Testing (The Product Manager)**
*   You deploy. 1% of users get the new model.
*   You track metrics that actually matter to the business: "Did they cancel their subscription?" "Did they copy-paste the code?"
*   If the model is smart (high MMLU) but rude (low EQ), users will cancel. Offline eval won't catch "rudeness," but A/B testing will.

### Section 3.4: Failure Analysis (The "Forensic Files" of AI)

Don't look at the 90% you got right. Look at the 10% you got wrong.
*   **Cluster the errors.**
    *   Are they all long queries? (Context Window issue).
    *   Are they all queries discussing politics? (Refusal/Safety issue).
    *   Are they all queries in Spanish? (Tokenization issue).
*   **Slice-Based Learning:**
    Divide your data into slices (e.g., Short vs Long, English vs French).
    If your accuracy is 95% overall, but 20% on French, you don't have a model problem. You have a French data problem.
*   **The "Feedback Sandwich":**
    The best evaluation loops are:
    1.  **Automated Filter:** Catch syntax errors.
    2.  **LLM Judge:** Catch obvious hallucinations.
    3.  **Human Expert:** Catch subtle logic flaws.
    4.  **Production A/B:** Catch user unhappiness.

---

## Tutorial 4: Autoregressive Models Beyond Language
*Or: Everything is a Sequence if You Squint Hard Enough*

### Section 4.1: The Unification

The core premise of the tutorial was the unification of modalities under the transformer architecture via **Next Token Prediction**.
*   **Vision:** From CNNs (Spatial) -> Vision Transformers (Patch Tokens).
*   **Audio:** From Spectrograms/RNNs -> Audio Tokens.
*   **Action:** From Continuous Controls -> Discretized Action Tokens.

This shift allows us to treat image generation not as a pixel regression problem, but as a sequence modeling problem, unlocking the scalability of LLMs for vision (e.g., in Video Generation).

### Section 4.2: The Physics of "Next Token Prediction"

The tutorial started with pure probability theory.

$$p(x) = \prod_{t=1}^{T} p(x_t \mid x_{&lt;t})$$

This formula says: The probability of an entire image is just the product of the probabilities of each pixel, given the previous pixels.

**Wait, what is a "previous" pixel?**
An image is a grid. It doesn't have a start and end.
**Convention:** Raster Scan. Top-left to bottom-right.
We force an order onto the image. We turn space into time.
Once we force an order, we can use GPT-4 to generate images.
*   *Critique:* Is Raster Scan optimal? Why not spiral? Why not Hilbert Curve?
*   *Answer:* It doesn't matter. Transformers are powerful enough to learn the pattern anyway.

### Section 4.3: Pixel vs Token (The discrete bottleneck)

*   **Attempt 1: PixelRNN (The Old Way, circa 2016)**
    Predict R, G, B for Pixel 1. Then Pixel 2.
    *   *Problem:* An iPhone photo is 12 megapixels. that is 12,000,000 steps.
    *   Transformers are \(O(N^2)\) in memory usage.
    *   \(12,000,000^2\) is... a number I cannot pronounce. The sun will burn out before you generate one selfie.

*   **Attempt 2: VQ-VAE / VQ-GAN (Vector Quantized Variational Autoencoder)**
    This is the magic unlock.
    1.  Take the 256x256 image (\(x\)).
    2.  Run a CNN Encoder to compress it into a 32x32 grid of "latent vectors."
    3.  **Quantize** these vectors: Snap them to the nearest entry in a fixed "Codebook" of 8192 learned vectors.
    4.  Now your image is a sequence of \(32 \times 32 = 1024\) integers (tokens).
    5.  Train a Transformer on these 1024 tokens.
    6.  *Success:* This is how standard image gen models work. You aren't predicting pixels. You are predicting "Visual Concepts" (Token #405 might mean "Texture of a dog's nose").

### Section 4.4: Video Generation (Sora explained)

If an image is a sequence of tokens, a video is just a longer sequence of tokens.
`[Frame 1 Tokens] -> [Frame 2 Tokens] -> [Frame 3 Tokens]`

**Why AR wins over Diffusion for Video:**
*   **Diffusion:** Generates the whole block at once (or iteratively denoises). It struggles with *temporal consistency*. The cat turns into a dog halfway through because the model "forgot" what the cat looked like 2 seconds ago.
*   **Autoregressive:** Has **Attention**. It looks back at Frame 1 while generating Frame 100. It has "perfect memory" (limited only by the context window).
*   *Result:* The cat stays a cat. The physics remains consistent.

### Section 4.5: Scaling Laws (The "Bitter Lesson")

They showed graphs that were deeply satisfying and deeply boring.
*   **Loss vs Compute:** Straight line (on log-log scale).
*   **Loss vs Data:** Straight line.
*   **Loss vs Model Size:** Straight line.

**Implication:** We haven't hit the wall.
In 2015, we tweaked architectures (ResNet vs Inception vs VGG) to get gains. We hand-designed features.
In 2025, we just add zeros to the budget.
If you 10x the compute, you get predictable gains in visual quality.
This turns AI from an "Art" (finding the right architecture) into "Industrial Engineering" (optimizing the supply chain of GPUs).
It is depressing for the artist, but great for the capitalist.

### Section 4.6: Inference Speed (The Pain Point)

Autoregressive models are **Serial.**
To generate token 100, you *must* have generated token 99.
You cannot parallelize time.
This makes them slow. Generating an image token-by-token takes 5-10 seconds. Diffusion can do it in 1 second.

**Technique: Speculative Decoding** 
This is a brilliant hack.

**The Setup:**
You have a Big Model (Slow, Smart) and a Small Model (Fast, Dumb).

**The Process:**
1.  The Small Model guesses the next 5 tokens: `[A, B, C, D, E]`. It takes 1ms.
2.  The Big Model checks all 5 guesses *in parallel* (because it has the inputs). It takes 10ms.
3.  If they are right, you just generated 5 tokens in 11ms (instead of 50ms).
4.  If token C is wrong, you keep A and B, discard the rest, and restart.

**Analogy:**
It's like having an intern (Small Model) write the email draft. The CEO (Big Model) just reads it and says "Good, Good, Good, No change this word." The CEO processes it much faster than writing it from scratch.

---

## Tutorial 5: Foundations of Imitation Learning
*Or: Why Robots Are Like Toddlers*

**Presenters:** Adam Block, Dylan Foster, Max Simchowitz

### Section 5.1: The Statistical Challenge of Imitation

The fundamental problem with simply "cloning" an expert is **Covariate Shift**.
When training a policy \(\pi_\theta(a|s)\) to mimic an expert \(\pi^*(a|s)\) using supervised learning (Behavioral Cloning), we assume the data is i.i.d.
However, in a sequential decision process, small errors accumulate.
*   If the trained policy makes a mistake with probability \(\epsilon\) at each step.
*   The state distribution \(P_{test}(s)\) drifts away from the training distribution \(P_{train}(s)\).
*   This leads to an error bound of \(O(\epsilon T^2)\) over a horizon \(T\). This quadratic compounding is why naive BC fails in long-horizon tasks.

### Section 5.2: Interactive Imitation Learning (DAgger)

To address the \(O(T^2)\) problem, we need to correct the distribution shift.
**DAgger (Dataset Aggregation)** creates a feedback loop:
1.  Train \(\pi_\theta\) on current dataset \(\mathcal{D}\).
2.  Run \(\pi_\theta\) to collect new trajectories.
3.  Ask the **Expert** to label the states visited by \(\pi_\theta\) (i.e., "What *would* you have done here?").
4.  Add these to \(\mathcal{D}\) and retrain.

This reduces the error bound to \(O(\epsilon T)\), linear in time.
*Critique:* DAgger is disjointed; humans cannot easily label states offline. **HG-DAgger (Human-Gated DAgger)** improves this by having humans intervene *only* when the robot is about to fail, reducing the labeling burden.

### Section 5.3: Offline RL and IQL

In many cases, we cannot query an expert (e.g., using historical medical data). We must learn from a static dataset that may contain suboptimal behavior.
**Implicit Q-Learning (IQL):**
Instead of cloning the average behavior (BC), IQL attempts to clone only the *best* behavior in the dataset.
*   It estimates a Value Function \(V(s)\) using expectile regression.
*   It filters the dataset to find actions where \(Q(s,a) > V(s)\) (actions better than average).
*   It performs weighted BC on these "advantageous" actions.
*   *Result:* The agent can outperform the average demonstrator in the dataset without ever interacting with the environment.

### Section 5.4: The LLM Connection

The tutorial drew a direct parallel between Robotics and LLMs:
*   **Pre-training:** This is Behavioral Cloning on the internet. It suffers from Covariate Shift (hallucinations lead to more hallucinations).
*   **RLHF:** This is effectively a variant of Offline RL (learning from preferences) or Interactive Imitation (on-policy generation + ranking).
*   **Chain-of-Thought:** This is "planning" in the latent space before executing an "action" (outputting the answer), allowing the model to correct its own distribution drift.

---

## Tutorial 6: Model Merging
*Or: The Franken-Model Era*

### Section 6.1: Linear Mode Connectivity (LMC)

The tutorial began by refuting the long-held belief that neural network weights are distinct, isolated local minima.
**The Discovery:**
If two models (A and B) share the same initialization (pre-training), and are fine-tuned on different tasks, there exists a linear path between them where the loss remains low.

$$\mathcal{L}(\alpha \theta_A + (1-\alpha) \theta_B) \approx \text{low}, \quad \forall \alpha \in [0, 1]$$

This **Linear Mode Connectivity** is the foundational property that enables all modern merging techniques. It implies that fine-tuning happens in a shared basin of the loss landscape.

### Section 6.2: Task Arithmetic (The Algebra of Skills)

This section introduced the concept of **Task Vectors**.
We can represent a "skill" as the difference between the fine-tuned weights and the base model weights:

$$\tau_{task} = \theta_{finetuned} - \theta_{base}$$

Surprisingly, these vectors behave algebraically:
*   *Addition:* \(\theta_{new} = \theta_{base} + \tau_{math} + \tau_{code}\) (Result: A model good at both).
*   *Negation:* \(\theta_{new} = \theta_{base} - \tau_{toxic}\) (Result: A model that reduces toxicity).
*   *Analogy:* This is similar to Word2Vec (\(\text{King} - \text{Man} + \text{Woman} = \text{Queen}\)), but applied to the weights of a 70B parameter model.

### Section 6.3: Interference Reduction (TIES and DARE)

Simply adding vectors often leads to **Interference**.
*   *Example:* The "Math" vector wants to increase weight \(w_{ij}\) by +0.5. The "Code" vector wants to decrease it by -0.3.
*   *Result:* An average of +0.1, which serves neither task well.

**TIES-Merging (Trimming, Interference Election, Sign)** 
1.  **Trim:** Remove the bottom 80% of smallest updates (noise).
2.  **Elect:** For each parameter, count the votes for positive vs negative usage.
3.  **Sign:** Keep only the updates that agree with the majority direction. Zero out the dissenters.
This "greedy" approach ensures that dominant changes are preserved without destructive interference.

**DARE (Drop And Rescale)** 
A stochastic approach to merging.
*   Randomly drop \(p\%\) of the delta weights (set to 0).
*   Rescale the remaining weights by \(1/(1-p)\).
*   This sparsifies the updates, reducing the probability of collision between multiple models.
*   *Result:* Empirically allows merging of up to 100+ fine-tuned models into a single backbone.

### Section 6.4: Evolutionary Merging (Model Soups)

Instead of manually picking coefficients (\(\lambda_1, \lambda_2\)), we can optimize them.
**Model Soups** runs a lightweight evolution on the validation set to find the optimal mixing coefficients for a set of models.
**Algorithm** 
1.  Sort models by validation accuracy.
2.  Add model \(M_2\) to \(M_1\). If Val Acc increases, keep. If not, discard.
3.  Repeat.

*   *Benefit:* Achieves SOTA accuracy without any additional training compute capability (inference only).

---

## Conclusion: The Maturity of the Field

As we wrapped up the tutorials, one feeling was inescapable: **We are growing up.**

The "Magic" phase is ending.
We are entering the "Engineering" phase.
*   We have XAI to inspect the machine.
*   We have Benchmarks to test the machine.
*   We have Evaluations to certify the machine.
*   We have Merging to assemble the machine from parts.
*   We have Autoregressive scaffolding to build the machine of any modality.

This is less exciting than "Magic," sure.
But Magic is dangerous. Engineering is reliable.
And after the year we've had in AI, I think we could all use a little more reliability.

Now, if you'll excuse me, I'm going to go merge a "Sarcasm Model" with an "Evaluation Benchmark" and see if it hurts the model's self-esteem.

*Tutorial materials and slides available at the NeurIPS 2025 website: https://neurips.cc/virtual/2025/tutorials*
