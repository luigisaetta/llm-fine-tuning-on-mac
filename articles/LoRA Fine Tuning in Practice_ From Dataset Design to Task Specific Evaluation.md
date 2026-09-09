# LoRA Fine Tuning in Practice: From Dataset Design to Task Specific Evaluation

This is the second article I dedicate to fine tuning a Small Language Model using LoRA.

In the first article I focused mainly on the fundamental mechanisms of fine tuning. In this second article, and in the accompanying demo, I decided instead to start from a more meaningful use case, a problem that is still simple enough to reproduce in an experiment, but that could also have a concrete relevance in an enterprise context.

## The use case: IT support ticket triage

Let us imagine that a company wants to improve its internal IT support service.

Every day, the support team receives a certain number of requests related to IT problems. Each request starts as a ticket and mainly contains free text in which the user describes the problem and the impact that problem is having on their work.

One of the first activities performed by the support team is ticket triage.

In particular, they need to identify the category of the problem, determine its severity and therefore help establish its priority, produce a summary of the problem, and route the ticket to the team with the most appropriate skills.

Let us also imagine that today this first classification is performed by a generalist team. The process works, but it can produce errors or inconsistencies. The category may be assigned incorrectly, severity may be evaluated inconsistently and, in some cases, the ticket may be routed to the wrong subteam, introducing delays in handling it.

The company therefore decides to experiment with an AI based system to make this process faster and more consistent.

## The task we want to assign to the model

The idea is to use a Small Language Model.

The model receives the ticket text, written in English, as input and must produce structured information as output, for example a JSON object containing three fields:

```json
{
  "category": "...",
  "severity": "...",
  "summary": "..."
}
```

The task therefore combines different capabilities.

The model must classify the ticket by assigning a category and a severity level among the allowed values, and it must also produce a short summary of the problem described by the user.

We also want the output to follow a well defined structure, producing valid JSON that is consistent with the required schema.

So the question we start from is simple:

**can fine tuning a Small Language Model be an appropriate solution for this use case?**

The goal of the article is not to prove that this is the only possible way to address the problem. A much larger model, given appropriate instructions, could probably perform a similar task even without fine tuning.

Here we want to explore a different possibility: specializing a much smaller model, compact enough to be trained and run on local hardware, with very limited costs.

## Fine tuning to learn a specific task

From a methodological point of view, we are in the area of supervised fine tuning and, more specifically, what we can consider task specific instruction fine tuning.

The goal is not simply to provide the model with new information. We want to modify its behavior so that, given an input with certain characteristics, it learns to perform a specific task reliably and produce the output in the format we have defined.

In our case, each training example will therefore contain the original ticket text as input and, as the expected output, the corresponding category, severity and summary.

The basic principle behind instruction fine tuning is exactly this: starting from an already pretrained model and specializing its behavior using examples that show what response we expect for a given input.

Our goal is therefore to see how well a relatively small model can learn this specific behavior.

## The role of data

The possibility of obtaining a good result depends to a large extent on the data available.

To train the model, we need a sufficiently representative set of historical tickets for which we know the expected result, meaning category, severity and a good quality summary.

But the amount of data is only part of the problem.

Its quality is probably even more important.

If categories were assigned inconsistently in the past, if severity levels were not clearly defined, or if very similar tickets received different classifications, the model will inevitably learn at least some of those inconsistencies.

The dataset, therefore, is not simply the material we use for training. To some extent, it also represents the specification of the behavior we want to teach the model.

In the article and in the demo, I will try to show that, with sufficiently consistent data, even a relatively small model, in our case one with about 1.7 billion parameters, can reach an interesting level of accuracy on a very specific task.

We are not looking for the perfect model, nor are we trying to prove that a 1.7B model is necessarily the best choice for a production system.

The goal is more concrete: to understand how far we can go by specializing a small model on a well defined problem.

## Another constraint: limited hardware resources

We deliberately introduce another constraint: fine tuning must be possible using relatively limited hardware resources.

In my case, the reference platform is the MacBook I normally use, equipped with 48 GB of unified memory. If possible, we will also see how the same approach can be moved to a relatively small GPU, for example an NVIDIA A10.

This choice is not accidental.

Many open weight models are available as families of models with different sizes. Qwen3, which we will use as our reference, is an example. Within the same family we can work with models of different sizes, including a 1.7 billion parameter version.

Even if, in a production system, we later decided to use a larger model, it can be convenient to develop the process initially using a smaller version.

This allows us to run experiments more quickly, work on the dataset, try different training configurations, validate the evaluation pipeline and correct problems with much lower time and cost.

There is also another aspect that I consider important.

A Small Language Model can be run locally or on relatively inexpensive infrastructure. For a company, this can mean very low inference costs, greater control over deployment, and the possibility of dedicating a model to a very specific task without having to use a much larger general purpose model for every single ticket.

## The educational goal

Finally, there is an educational reason.

One of the goals of this article is to make the experiment reproducible without necessarily requiring expensive infrastructure.

I would be happy if, in addition to practitioners, this journey could also be useful to university students or people at the beginning of their AI careers who want to understand in practical terms what it means to fine tune a Language Model.

To really understand some of these concepts, it is useful to experiment with them directly.

Preparing the dataset, configuring LoRA, starting the training process, observing the loss, evaluating the model and analyzing its errors provide a very different level of understanding compared with simply using a model through an API.

And this is exactly the path we will follow in the demo.

## Preparing the datasets

The first important task is to prepare the datasets.

If we want to do things properly, we start with the idea of having, in the end, three separate and verified datasets: one for training, one for validation and one to be used, at the end of the training phase, for testing.

The validation dataset is used during training. In our case, at the end of each epoch, we will use it to evaluate the results obtained by the model. It will also help us decide, among the checkpoints corresponding to the different epochs, which one to use for the final test.

The test dataset, instead, is NOT used during training and is not used to choose the checkpoint. We will use it only at the end, as a kind of final acceptance test.

If the results are not satisfactory, we can of course decide to iterate again. From a rigorous point of view, however, if we repeatedly use test results to modify our choices, the test dataset also starts to influence the development process indirectly. In a real project, after several iterations, it would therefore be appropriate to have a new dataset available for the final test.

If you want my opinion, also based on my experience, at the beginning of a project like this I would comfortably assume that around 60 percent of the overall effort will be dedicated to building and validating the data.

We have already said that the model also learns the inconsistencies present in the datasets. We need to add another important aspect to this: the distribution of the examples.

If some categories are underrepresented, it is reasonable to expect the model to have more difficulty precisely on those categories. In our case, we also need to check the distribution of severity levels and of category, severity pairs.

This does not necessarily mean having the same number of examples for every possible combination. Some combinations may be uncommon or unrealistic. But it does mean knowing what data we are using to train the model.

For readers with less experience in this area, this is an important point.

At this stage, knowing LLMs and the mechanics of fine tuning is NOT enough. You also need the attention to data quality and distribution that has always been part of the Data Scientist's job.

## How do we proceed in our case?

In our case we do not have real tickets available. We therefore want to build a synthetic dataset.

I started by defining and documenting a set of rules, and then used those rules to generate datasets that followed them as much as possible.

The domain, as we said, is IT support ticket classification.

I therefore started by defining 10 categories. For each category, I wrote a short description and some rules that help determine when a ticket belongs to that category.

I then defined four severity levels:

`P1`, `P2`, `P3`, `P4`

in decreasing order of severity.

I also defined rules for severity, so that the assignment of P1, P2, P3 or P4 would not be arbitrary.

At this point we need to remember one of the constraints we set for ourselves: limited compute resources.

I therefore assumed, as a starting point, that three relatively small datasets could be sufficient.

The training dataset contains 1,000 examples, meaning 100 examples for each category.

The validation dataset contains 200 examples.

The test dataset contains another 200 examples.

I am not saying that 1,000 examples is the ideal number for this problem. It is simply an initial choice, compatible with the kind of experiment I want to run and with the constraint of working on local hardware.

## Generating the datasets

After writing a first version of the rules, I used an AI Assistant, in my case ChatGPT with GPT 5.6, to help me with two activities.

The first was reviewing and correcting the rules, looking for ambiguities and inconsistencies, without trying to make them artificially perfect.

The second was generating the three datasets, asking in particular to avoid underrepresented categories and to respect, as much as possible, the rules I had defined.

Of course, I did not simply take the generated files and use them directly.

I performed a manual review, although only partial, identified some problems, requested some changes and eventually arrived at version V1 of the three datasets.

This is the version from which the experiment starts.

One thing to keep in mind is that we are working with synthetic data. Real tickets could be noisier, less regular, contain incomplete information, errors, abbreviations and difficult cases that our generation process may not fully reproduce.

For the purpose of the demo this compromise is acceptable. But it is important to remember it when we analyze the results.

As anticipated, all the code for the demo is available in my GitHub repository:

[https://github.com/luigisaetta/llm-fine-tuning-on-mac](https://github.com/luigisaetta/llm-fine-tuning-on-mac)

The code related to this use case is in the `ticket-classification` directory.

Version V1 of the datasets is available under `artifacts/datasets`.

Later we will see that this will not be the final version of the data. And, in my opinion, this is actually one of the most interesting parts of the experiment.

## The fine tuning approach

After creating V1 of the datasets, and performing a first set of checks, I moved on to defining the fine tuning part of the process.

The model I selected is a Small Language Model, Qwen3 1.7B.

Its size allows it to run in BF16 using only a few GB of memory, approximately 4 or 5 GB depending on the implementation and inference conditions.

Here, however, we need to pay attention to an important point.

If we simply want to run a model, meaning inference, a first estimate of the amount of memory required to hold the weights is quite simple.

In BF16 each parameter requires 2 bytes. A model with approximately 1.7 billion parameters therefore requires around:

\[
1.7 \times 10^9 \times 2 \simeq 3.4 \text{ GB}
\]

This, however, is only the memory required for the weights.

During inference, additional memory is required, for example for the KV cache and for some intermediate activations. The actual amount also depends on context length, the number of generated tokens and the implementation being used.

On Apple architecture the situation is interesting because memory is unified. CPU and GPU share the same memory.

Therefore, even if on a MacBook like mine the number of GPU cores is relatively limited, I can have access to a significantly larger amount of memory than what is available on some discrete GPUs.

On an Intel or AMD machine with an NVIDIA GPU, instead, you need to look mainly at the VRAM available on the GPU, which can be much smaller than system RAM, for example 8 GB.

But all of this mainly concerns inference.

Fine tuning requires more memory.

During training, we also need to perform backpropagation and keep additional information in memory, including the activations required to compute gradients, the gradients themselves for the parameters we are training and the optimizer state.

If we performed full fine tuning, we would need to update all the model weights, in our case about 1.7 billion parameters.

This is where Parameter Efficient Fine Tuning, PEFT, comes into play.

The general idea is simple: instead of training all the parameters of the model, we make only a small fraction of them trainable, while keeping the original model weights frozen.

To make the idea concrete, imagine going from about 1.7 billion parameters to only a few million trainable parameters.

The reduction in the additional memory required for gradients and optimizer state becomes very significant.

Of course, an important question remains: will such a small number of trainable parameters be sufficient to adapt the model to our task?

It depends on the complexity and variety of the task.

In our case, we will only know the answer at the end of the experiment.

## LoRA

PEFT actually identifies a family of techniques.

The one we will use is LoRA, Low Rank Adaptation.

With LoRA, we do not directly modify the original model weights. We keep those weights frozen and introduce a small number of new trainable parameters into selected Transformer modules.

Let us try to understand the idea without going too deeply into the mathematics.

Imagine we have a matrix \(W\) containing the weights of one of the modules to which we want to apply LoRA.

After fine tuning, we would like to obtain something equivalent to:

\[
W_{updated} = W + \Delta W
\]

The matrix \(W\) can be very large.

As a deliberately simple example, imagine a matrix with dimensions:

\[
10000 \times 10000
\]

We are talking about 100 million parameters.

The trick used by LoRA is not to learn all the values of the matrix \(\Delta W\) directly.

Instead, LoRA represents \(\Delta W\) as the product of two much smaller matrices:

\[
\Delta W = BA
\]

For example, we could have:

\[
A: 16 \times 10000
\]

and:

\[
B: 10000 \times 16
\]

The product still produces a matrix with dimensions \(10000 \times 10000\), but the number of parameters we actually need to train is:

\[
16 \times 10000 + 10000 \times 16 = 320000
\]

So, in this example, we have moved from 100 million possible parameters in the update to 320 thousand trainable parameters.

The value 16 is what LoRA calls the rank.

Of course, this is a simplified representation. In the real model, LoRA is applied to specific Transformer modules, and we need to decide which modules to use and which rank to choose.

But the fundamental idea is this: we are imposing the constraint that the model adaptation can be represented through low rank matrices.

For more information about LoRA, my personal suggestion is once again Sebastian Raschka's book mentioned earlier. In particular, Appendix E provides a very clear explanation of the method and its implementation.

## Do I really need to remember all that algebra?

The previous details may have worried some readers, especially those who no longer remember perfectly the matrix algebra they studied during the first years of university.

The good news is that you do not need to implement all of this from scratch.

If you really want to understand what happens during fine tuning, I think it is useful to understand at least the basic idea behind LoRA and the meaning of its main parameters.

From the coding point of view, however, the work is now enormously simplified by the availability of excellent Python libraries.

In our case we will use the Open Source libraries provided by Hugging Face, in particular PEFT, which allow us to configure and apply LoRA with only a few lines of code.

This allows us to focus on the decisions that really matter for our experiment: which modules to adapt, which rank to use, which hyperparameters to choose and, above all, how to understand whether the model is actually learning the task.

## The code for implementing fine tuning with LoRA

As I said before, today there are excellent libraries that have simplified and, to a large extent, standardized the way we can implement Supervised Fine Tuning using PEFT and LoRA.

A first possible approach is quite simple: take a good working example and adapt it to your own case, especially to your datasets.

In our case you can directly use the examples available in my GitHub repository, in the `ticket-classification` directory.

Of course, you still need to know what you are doing.

My suggestion is therefore to study at least the main details of the code and understand the meaning of the configurations you are using.

One of the first things to pay attention to is that we want to run training on the GPU.

On Macs with Apple Silicon, PyTorch provides the MPS backend, Metal Performance Shaders, which allows us to use the Apple GPU. You can find the corresponding settings in the demo code.

If instead you want to run training on Linux using an NVIDIA GPU, you need a Python environment in which PyTorch and CUDA support are correctly installed and configured.

I will not go into installation details here, also because they depend on the hardware, operating system and library versions you are using. The important point is always to verify that PyTorch is actually using the GPU and is not running the training on the CPU.

## Getting help from a Coding Agent

There is also another possibility, which today has become very practical: getting help with the code from a Coding Agent, for example Codex or Claude Code.

The documentation for the main libraries is now very good and available online, so these tools can be very effective in building a first version of the training pipeline.

My suggestion, however, is to use them with some discipline.

First of all, provide the agent with sufficiently precise specifications, so that the resulting code follows a style you understand and feel comfortable with.

Then review the code.

If you simply keep adding generated code without checking it, after a few iterations you risk ending up with a project that may work, but becomes increasingly difficult to understand and modify.

In my case I used Codex to help me develop this demo.

However, I had already written this kind of code in the past. I think my first fine tuning experiments date back about three years, when I worked directly from examples and documentation. This helps me know which points need to be checked and, above all, understand the meaning of the configurations being used.

## Epochs, checkpoints and loss

There is another important aspect we need to consider.

Fine tuning typically proceeds through multiple passes over the training dataset.

A complete pass through the entire training dataset is called an `epoch`.

We can therefore decide to train for several epochs and save a model checkpoint at the end of each one.

But it is absolutely not guaranteed that more epochs will always produce a better model.

The best checkpoint may not be the one from the final epoch.

For this reason, it is important to monitor at least the training loss and the validation loss.

Training loss measures the model's behavior on the data used for training, while validation loss is calculated on data that is not used to update the parameters.

If during training the first keeps decreasing while the second starts increasing, this can be a sign of overfitting.

I deliberately wrote "can be".

A single curve is not always enough to understand what is really happening, especially if our task has characteristics that we can measure more directly.

And this is exactly what happened in our experiment.

Let us see what I observed during the first training run using V1 of the datasets.

## The first experiments and the need for additional metrics

The first training run using version V1 of the datasets produced exactly what I did not want to see.

From the very first epoch, training loss started decreasing while validation loss started increasing.

If I had stopped at the theory usually taught in an introductory course on neural network training, I might have concluded: with this setup, there is no point in continuing for more epochs.

And perhaps, incorrectly, I would have started changing the learning rate or the LoRA hyperparameters.

Nothing could have been more wrong.

First of all, we need to remember that neural networks of this size and capability are systems with complex, nonlinear dynamics. Making changes without having at least a hypothesis about what is happening is, in my opinion, not a good strategy.

The second point, again as my personal opinion, is that whenever possible I prefer to look at the data first.

But, going back to the first point, we need some indication of why changing the data might be useful.

So let us start from the symptoms.

What we have observed is simply a validation loss trend that we do not like.

But what does that trend tell us about the model's ability to learn the task we care about?

Not enough.

To understand why, we need to go a little deeper into how loss is calculated during training.

In simplified terms, during training the model must predict the next token in the sequence. The loss measures how consistent the probability distribution produced by the model is with the expected token.

It is a fundamental training metric, but it is still quite far from the type of question that matters to us.

We do not simply want to know whether the model has become better at predicting the token sequence in the validation set.

We want to know whether it has learned our task.

So let us add our understanding of the problem.

The model must do at least three things.

1. It must generate valid JSON, with the fields `category`, `severity` and `summary`.

2. It must identify the correct category, choosing among only 10 possible values.

3. It must identify the correct severity, choosing among 4 possible values.

Then there is the summary.

For the moment, let us leave it aside, because evaluating it requires a different discussion.

At this point, after consulting my AI Assistant as well, what I sometimes call my "co intelligence", before changing the datasets and, even worse, before starting to change the hyperparameters, I decided to modify the code.

I added the calculation, at the end of each epoch, of several metrics directly related to the task:

1. `Category Accuracy`

2. `Severity Accuracy`

3. `Joint Category and Severity Accuracy`

4. `Valid JSON Accuracy`

The third metric is particularly easy to interpret: we consider the result correct only when both category and severity are correct.

The fourth measures the percentage of outputs that can be correctly parsed as JSON according to the expected structure.

Hugging Face libraries allow us to extend the training loop, for example using callbacks and custom evaluation logic, so that we can calculate and display metrics specific to our task.

I think this is an important point.

Standard training metrics are necessary, but they are not always sufficient. If our problem has a precise structure, we should try to measure directly what actually matters to us.

At this point I ran a second experiment, still using version V1 of the datasets, but this time collecting these additional metrics.

For your curiosity, on my MacBook, with Qwen3 1.7B and datasets of this size, 8 epochs require slightly less than 50 minutes.

For me, this is a perfectly reasonable amount of time for a single experiment.

And it was by looking at these new metrics that the situation started to become much more interesting.

## The second run and what the new metrics tell us

The result of the second run was, in my opinion, illuminating, precisely because metrics directly related to the task allowed me to "look inside" and better understand what was happening.

Here are the numbers, epoch by epoch.

| Epoch | Train loss | Val loss | Category | Severity | Cat + Sev | JSON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0579 | 0.8048 | 80.0% | 39.5% | 32.5% | 100% |
| 2 | 0.0527 | 0.9429 | 81.0% | 42.0% | **36.5%** | 100% |
| 3 | 0.0397 | 1.0455 | 79.5% | 40.0% | 34.0% | 100% |
| 4 | 0.0421 | 1.0146 | **82.5%** | **43.0%** | **36.5%** | 100% |
| 5 | 0.0408 | 1.1329 | 80.5% | 34.0% | 31.5% | 100% |

If we look at these numbers, we immediately notice some positive aspects.

The first is that, already after the first epoch, the model has learned to generate valid JSON according to the required structure. Accuracy remains at 100 percent during the following epochs as well.

The second is that the model is reasonably good at predicting the category. We remain consistently around 80 percent, with a maximum of 82.5 percent at epoch four.

So the story is already different from what we might have concluded by looking only at validation loss.

Continuing training for multiple epochs is not completely useless.

There is another interesting result, although it requires a little attention.

The joint accuracy of category and severity is quite close to the product of the two individual accuracies.

For example, at epoch four we have:

\[
0.825 \times 0.43 \simeq 0.355
\]

while the observed joint accuracy is:

\[
0.365
\]

They are not exactly the same, but they are very close.

This suggests that, at least at this stage, category errors and severity errors do not appear to be strongly correlated.

However, we should not confuse this with another property of the dataset.

In the rules I defined, category and severity were designed as two distinct dimensions of the problem. But the fact that joint accuracy is close to the product of the two accuracies concerns the independence of the events "category is correct" and "severity is correct", not directly the independence of category and severity as domain variables.

That said, the message remains positive for me.

The model is clearly learning something.

It has learned the output format.

It has learned to classify the category reasonably well.

And the metrics finally allow us to see where the real problem is.

## The problem is severity

The model is not learning how to evaluate severity equally well.

Accuracy remains around 40 percent.

The result is not completely negative. With four possible severity levels, a naive classifier choosing randomly, assuming a uniform distribution, would have an expected accuracy of 25 percent.

So the model is certainly learning something.

The problem is that we cannot get much beyond approximately 40 percent.

And, above all, we do not see any significant improvement as we continue with more epochs.

At this point the problem looks much more localized: it is not that the model is unable to learn the task in general, it is mainly severity classification that is not working well enough.

I therefore performed another analysis, looking at the confusion matrix for `P1`, `P2`, `P3` and `P4`.

The result clearly showed that the model tended to predict mainly `P2` and `P3`.

At this point we had several clues pointing in the same direction.

The model was learning the JSON format well.

It was learning the category reasonably well.

It was having much more difficulty with severity and tended to concentrate on the central classes.

Before changing the learning rate, LoRA rank or other hyperparameters, the question for me became much simpler:

**is there something wrong with the data we are using to teach severity?**

All the clues were starting to suggest that there was.

## Dataset analysis and V2

At this point the analysis moved to the datasets.

In particular, I tried to understand two things.

The first was whether there were significant differences between the training set and the validation set.

The second was whether there were specific problems in the distribution and definition of severity.

For this analysis I used a combination of scripts, with deterministic metrics and checks, and a long interactive session with ChatGPT to examine the data, look for suspicious patterns and compare examples belonging to the different datasets.

Summarizing as much as possible, two main problems emerged.

The first concerned the distribution of severity levels in the training dataset.

For some categories, one severity level was clearly dominant and, in some cases, practically the only one present.

This is clearly a problem.

If, for example, all or almost all examples in category `C1` have severity `P2`, the model can learn a very simple shortcut:

`if category = C1, then severity = P2`

But this is not the behavior we want to teach.

We want severity to be determined using the information contained in the ticket, especially the impact of the problem, and not simply inferred from the category.

The second problem concerned the relationship between training and validation.

Some examples in the validation set were more difficult to interpret, especially with respect to severity, than the majority of examples in the training set.

In other words, we were asking the model to generalize to cases with a level of complexity that was not sufficiently represented in the data used to teach it the task.

At this point we finally had a reasonably concrete hypothesis for why severity accuracy was stopping at around 40 percent.

Based on these findings, I gave ChatGPT precise instructions to generate a V2 of the training and validation sets, trying to reduce the problems we had identified.

In particular, I asked for a more appropriate distribution of severity levels within the categories and greater consistency between the complexity of the examples in the training set and those used for validation.

I deliberately chose NOT to modify the test set.

The test set therefore remained the V1 version.

This choice would later allow us to see the effect of the work done on the data without also modifying the dataset used for the final evaluation.

Of course, in a real scenario, the problem is more complex.

Training, validation and test data must be statistically representative of the real data on which the model will actually operate in production.

In particular, the tickets used during training should not be systematically simpler, cleaner or less ambiguous than the real tickets the model will receive once deployed.

And this is exactly where building synthetic datasets becomes difficult.

Generating many synthetic examples is relatively easy.

Generating synthetic examples that accurately represent the variety, ambiguity, errors, edge cases and distribution of real data is a much harder problem.

Remember what I said earlier about spending 60 percent of the time on the data?

At this point in the experiment, we are starting to understand why.

## Training on the V2 datasets

Before starting a new training run, there was one more improvement to make to the training code.

We concluded that validation loss alone tells us relatively little about whether the model is actually improving on the task we care about.

For this reason we introduced and evaluated, at the end of each epoch, several task specific metrics: Category Accuracy, Severity Accuracy, Joint Category and Severity Accuracy and Valid JSON Rate.

At this point a fairly natural question arises.

If we train for multiple epochs and save a checkpoint at the end of each one, how do we decide which checkpoint is the best?

The choice I made was to use `Joint Category and Severity Accuracy` as the main metric.

The best checkpoint would therefore be the one corresponding to the epoch with the highest value of this metric on the validation dataset.

Since we had already implemented the calculation of this metric at every epoch, the code change was simple. We only needed to specify this metric in the trainer configuration as the criterion for selecting the best model.

At this point I had everything I needed.

Training dataset V2, validation dataset V2, the new metrics and a criterion better aligned with our task for selecting the checkpoint.

So I started a new run.

About 50 minutes later, I had these results.

| Epoch | Training Loss | Validation Loss | Category Accuracy | Severity Accuracy | Category + Severity | Valid JSON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.051100 | 0.103894 | 78.5% | 99.5% | 78.5% | 100.0% |
| 2 | 0.002500 | 0.128145 | 85.0% | 100.0% | 85.0% | 100.0% |
| 3 | 0.000400 | 0.161117 | 86.5% | 100.0% | 86.5% | 100.0% |
| 4 | 0.000300 | 0.144557 | 87.0% | 100.0% | 87.0% | 100.0% |
| 5 | 0.000200 | 0.147938 | 87.0% | 100.0% | 87.0% | 100.0% |
| 6 | 0.000300 | 0.147822 | 86.5% | 100.0% | 86.5% | 100.0% |
| 7 | 0.000300 | 0.148585 | 87.0% | 100.0% | 87.0% | 100.0% |
| 8 | 0.000300 | 0.148948 | **88.0%** | **100.0%** | **88.0%** | **100.0%** |

In my opinion, these numbers tell a very interesting story.

Let us start once again from validation loss.

The best value is the one from the first epoch, `0.103894`.

After that, validation loss increases and remains higher for the rest of the training process.

If we had used only this metric to select the checkpoint, we would therefore have selected epoch one.

But let us look at what happens to the metrics that directly describe the task.

Category Accuracy increases from 78.5 percent in the first epoch to 88 percent in the eighth.

Severity Accuracy is already 99.5 percent at the first epoch and reaches 100 percent from the second.

The model also continues to generate valid JSON in 100 percent of the cases.

As a result, Joint Category and Severity Accuracy also increases from 78.5 percent to 88 percent.

According to the criterion we selected, the best checkpoint is therefore the one from epoch eight.

I think this result confirms an important point that already emerged from the previous run.

Validation loss remains an important metric, but it does not necessarily correspond to model quality on the specific application task.

In this case the difference is quite clear.

If we had selected the checkpoint using validation loss, we would have chosen epoch one, with a Joint Accuracy of 78.5 percent.

Using the task specific metric instead, we reach epoch eight, with a Joint Accuracy of 88 percent.

That is almost 10 percentage points of difference.

There is also a second very clear result.

The problem we had observed with severity seems to have almost disappeared. With V1, accuracy was around 40 percent, while with V2 it reaches 99.5 percent after the first epoch and 100 percent from the second.

This is an important change and, above all, we obtained it without changing the base model or the LoRA configuration.

Of course, a 100 percent validation result requires some caution. We may have built a validation dataset that is too simple, or too close to the distribution of the training set.

And this is where the decision we made earlier becomes important: we did not modify the original test dataset.

## Let us stop for a moment

We have done, and read, quite a lot.

I think it is useful to stop for a moment, if only to appreciate the results of the work done so far. And perhaps to find the energy to continue.

In summary, we understood that the main area on which we needed to focus our effort was the data.

Without changing the base model and without modifying the LoRA configuration, on the V2 validation set we reached:

`88%` Category Accuracy

`100%` Severity Accuracy

`88%` Joint Category and Severity Accuracy

`100%` Valid JSON Rate

We also saw that, at least in our experiment, continuing training up to 8 epochs was useful. If we had stopped after the first epoch, guided only by validation loss, we would have selected a worse checkpoint according to the metrics that actually matter to us.

That 100 percent severity accuracy probably also suggests that, from this point of view, the V2 validation set has become too simple or too regular compared with the training set.

For the moment I consider this a second order problem.

We now have a model that seems to have learned the task well on the data used during development.

But we must now remember that there is still one dataset that we have not used.

The test dataset.

## The test dataset

As I said earlier, I deliberately chose not to modify the test set when we moved from V1 to V2 of the data.

So this dataset did not benefit from the corrections we introduced in the training and validation sets.

At this point we can finally use it.

Once again, I will show the results first and then we can try to understand what they are telling us.

| Metric | Test score |
| --- | ---: |
| Category Accuracy | 90.5% |
| Severity Accuracy | 55.5% |
| Category and Severity Accuracy | 50.5% |
| Valid JSON Rate | 99.5% |

The first reaction might be: we have a problem again.

And yes, there is definitely something to understand.

But, in my opinion, these results are even more interesting than a test that had simply confirmed the numbers from the validation set.

Category Accuracy is actually 90.5 percent, slightly higher than what we observed on validation.

The ability to produce valid JSON also remains practically perfect, at 99.5 percent.

The problem, once again, is severity.

On the V2 validation set we had reached 100 percent.

On the original test set, instead, we reach only 55.5 percent.

As a consequence, Joint Category and Severity Accuracy drops to 50.5 percent.

So the test set is telling us something very specific.

The model has learned to recognize the category well and has learned the output format very well.

For severity, however, the behavior we obtained on the V2 validation set does not generalize nearly as well to the original test set.

And this is where the fact that we did not modify the test dataset becomes particularly useful.

If we had modified training, validation and test at the same time using the same criteria, we would probably have obtained much nicer numbers.

But we would also have lost an important piece of information.

The original test set is showing us that there is still something to understand in the way we defined, represented or taught severity.

## Comparing the model with the baseline

There is one more important check to perform.

So far we have evaluated the fine tuned model in absolute terms. To better understand the effect of fine tuning, we can compare it with the original Qwen3 1.7B model, without applying the LoRA adapter.

I performed this comparison in the notebook `demo07_ticket_classification_test_evaluation.ipynb`, using the same test dataset and the same evaluation procedure.

| Metric | Base model | Fine tuned adapter | Delta |
| --- | ---: | ---: | ---: |
| Category Accuracy | 0.0% | 90.5% | +90.5% |
| Severity Accuracy | 0.0% | 55.5% | +55.5% |
| Category and Severity Accuracy | 0.0% | 50.5% | +50.5% |
| Valid JSON Rate | 0.0% | 99.5% | +99.5% |

The result is quite clear.

With the type of input expected by our application, the base model does not directly perform the task, while the fine tuned model has learned the behavior we wanted to teach it.

However, this comparison must be interpreted correctly.

We provide the base model with essentially the same input as the fine tuned model, meaning the ticket text. We do not give it an extended prompt containing the category descriptions, the severity rules and some examples.

So we are not comparing fine tuning and prompting in general.

What we want to verify is simpler: the behavior observed after training was not present in the original model.

It was learned through fine tuning with LoRA.

The notebook used for this evaluation is available here:

[demo07_ticket_classification_test_evaluation.ipynb](https://github.com/luigisaetta/llm-fine-tuning-on-mac/blob/main/ticket-classification/demo07_ticket_classification_test_evaluation.ipynb)

## Let us look at a concrete example

Metrics are important, but at some point it is also useful to look directly at what the model produces.

In the last cell of the notebook `demo07_ticket_classification_test_evaluation.ipynb`, you can select a record from the test dataset and run the fine tuned model, displaying the original ticket, the generated response and the expected response together.

Let us look at an example.

**Ticket**

```text
The right side of a corporate laptop screen is physically damaged, making several application controls hard to use.
```

The model produces:

```json
{
  "category": "DEV-01",
  "severity": "P3",
  "summary": "Corporate laptop screen hardware is failing"
}
```

The expected response in the test dataset is:

```json
{
  "category": "DEV-01",
  "severity": "P3",
  "summary": "Corporate laptop display is physically damaged"
}
```

The generated JSON is valid.

In this case, both category and severity are correct.

The summary does not exactly match the one in the dataset, but it expresses substantially the same information.

This example also reminds us of an important characteristic of the task.

For `category` and `severity`, we can define very easily whether the answer is correct or not, because both fields must take one of the predefined values.

For the `summary`, the situation is different.

Two texts may use different words and still both be correct.

For this reason, up to this point we have deliberately left the summary out of our main metrics.

We will need to decide separately how to evaluate it.

## Some conclusions

The article is not short. It has probably become longer than I initially expected.

From my point of view, however, the experiment shows a few things quite well.

First, it is possible to specialize a Small Language Model with LoRA and obtain good performance on a specific task while working with deliberately limited hardware. In my case, all the training was performed on my MacBook, with training times short enough to allow several iterations.

The second point is exactly this: the process is iterative.

We cannot expect to define everything correctly on the first attempt. Some things only become visible when we run experiments, analyze errors and try to understand where to intervene.

In our case, much of the improvement came from the data.

We did not change the base model and we did not modify the LoRA configuration. We identified some problems in the datasets, built a V2 and obtained a significant improvement, especially on severity.

Finally, we saw the importance of task specific metrics.

Validation loss remains important, but by itself it did not allow us to understand what the model was actually learning. Category Accuracy, Severity Accuracy, Joint Category and Severity Accuracy and Valid JSON Rate gave us much more useful information for guiding the next decisions.

## We did not do everything

Deliberately.

The goal of this article was not to cover every possible aspect of fine tuning, but to show a complete and reasonably realistic process, from the dataset to the final evaluation.

There are at least two aspects that deserve further investigation.

The first is the significant difference we observed in severity between the validation and test datasets.

On the V2 validation set we reach 100 percent accuracy, while on the original test set we stop at 55.5 percent.

This difference definitely deserves further analysis. Once again, I would start from the data, trying to understand which differences exist between the two datasets, which types of examples produce errors and whether it is necessary to create a V3.

The second point concerns the summary.

Looking at several inference examples, I noticed that the summary produced by the model is often correct in meaning, but rather generic.

For example, a response such as:

`hardware XX is failing`

may be acceptable, but it is clearly less informative than a summary that describes the original problem more precisely.

This aspect may also contribute to the behavior we observe in validation loss, although demonstrating that would require a more specific analysis.

Here the problem also becomes more complex.

For category and severity we have a finite set of possible answers, so we can easily measure accuracy.

For the summary, instead, we want the model to understand a text and produce a good summary of it. Quality cannot be measured simply by comparing the generated string with the one stored in the dataset.

And, in this case, working only on the data may not be sufficient. We may discover that obtaining better summaries also requires a base model with stronger language capabilities.

This is one of the questions I would deliberately leave open.

Probably, if my work and sport commitments allow it, I will come back to some of these aspects in a future article.

For the moment, I will stop here.

I will add a technical appendix, however, for readers who want to go a little deeper into some of the choices made in the LoRA configuration.

## Appendix: LoRA, some additional information

LoRA is one of the Parameter Efficient Fine Tuning techniques.

It is probably one of the best known and most established, but it is not the only one. Other techniques exist, for example DoRA, which I may explore in a future article.

For obvious reasons, I have not gone into a detailed discussion of how LoRA works.

If you want to learn more, my first suggestion is once again Sebastian Raschka's book, *Build a Large Language Model (From Scratch)*. In particular, Appendix E contains a detailed explanation of LoRA and its implementation.

Another useful reference, by the same author, is this online article:

[LoRA and DoRA from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)

Finally, I want to add one note about the configuration used in my code.

One of the choices we need to make when using LoRA is which model modules it should be applied to.

In my case, I chose to apply LoRA to all the linear projections in the attention modules, meaning the matrices associated with query, key, value and output.

This is a fairly common choice. Of course, it is not the only possible one. We could decide to apply LoRA only to some projections, or extend it to other linear modules in the Transformer.

This choice, together with the rank and the other parameters in the LoRA configuration, determines the total number of trainable parameters and can influence both the cost of training and the model's ability to adapt to the task.

In our experiment, I preferred to keep this configuration fixed.

The goal, as we have seen throughout the article, was not to search for the optimal LoRA configuration through a long series of hyperparameter experiments, but to understand how much we could achieve by focusing mainly on data quality and model evaluation.