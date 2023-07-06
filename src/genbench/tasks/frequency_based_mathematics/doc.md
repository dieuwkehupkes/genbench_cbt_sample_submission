## Motivation
This example task measures memorisation by computing to what extent the accuracy a model achieves
on a mathematical reasoning task is dependent on the frequencies of the terms in the problem.
This measure is inspired by the work of [Razehgi et al (2022)](https://aclanthology.org/2022.findings-emnlp.59/),
who measure the correlation between the pretraining term frequencies and a models performance.
In this example task, we only look at multiplication, and define a generalisation score as 
`1 - abs(cor(accuracy, pretraining_term_frequencies)`. We compute the accuracy for a particular term
by multiplying it with 30 other terms, and computing the average accuracy. 

TODO: explain this a bit better

## Examples

The examples in this task are 

## Usage

Give an example on how the dataset can be used.

## Data Source
*Describe the data source for this Frequency based mathematics.*

## Limitations and Bias
*Note any known limitations or biases that the Frequency based mathematics has, with links and references if possible.*

## Citation
*Cite the source where this Frequency based mathematics was introduced.*

## Further References
*Add any useful further references.*
