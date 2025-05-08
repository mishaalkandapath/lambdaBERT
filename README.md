# Neural Logical Form Semantics
We describe a decoder for BERT that converts contextualized BERT vectors into semantic terms from the language of the lambda calculus.  While our results are in many respects not competitive yet with current semantic parsing work that uses SQL templates, it is a strong enough start to demonstrate that this problem is not intractable.  This work is important because it links current research in neural language modelling to a traditional notion of logical semantics, and can therefore address lingering questions about whether neural language models indeed learn semantics in a classical sense.
<p align="center">
<img width="30%" src="https://github.com/mishaalkandapath/lambdaBERT/blob/main/lambdafigs/modelschematic.png">
</p>

## Reproducing Results
### Setup
The easiest way to install all necessary packages would be to build a conda/virtualenv environment from the yaml file
#### Conda
```
conda env create -f environment_droplet.yml	# create & install
conda activate supertagger # activate
```
**Note**: this module requires pyClarion, which needs to be independently downloaded from [here](https://github.com/cmekik/pyClarion/tree/v2409)
After clone pyClarion and before installing in edit mode, do the following to apply project-specific patches to pyClarion:
 ### Running
Coming soon

### Some Results
<p align="center">
 <img width="50%" src="https://github.com/mishaalkandapath/lambdaBERT/blob/main/lambdafigs/confusion_matrix_test.png"><br>
 Confusion matrix on the four-way classifier demonstrates a near-perfect performance at the lowest level of granularity. This suggests that a basic level of understanding of the structure of the underlying semantic terms has been achieved.<br>
  
<div align="center">

| **Measure** | **Applications** | **Abstractions** | **Both** |
|:-----------:|:----------------:|:----------------:|:--------:|
| F-score     | 0.865            | 0.916            | 0.734    |
| Recall      | 0.841            | 0.913            | 0.723    |
| Precision   | 0.906            | 0.938            | 0.761    |

</div>

<div align="center">
  <img src="https://github.com/mishaalkandapath/lambdaBERT/blob/main/lambdafigs/test_f_scores_all.png" alt="Image 1" width="45%" style="display: inline-block; margin: 0 10px"/>
  <img src="https://github.com/mishaalkandapath/lambdaBERT/blob/main/lambdafigs/test_f_scores_abs.png" alt="Image 3" width="45%" style="display: inline-block; margin: 0 10px"/>
Accounting for either abstraction-based or application-based errors in F-scores does far better, 85-90% as opposed to ~75%. This suggests a somewhat complementary distribution of errors: sentences that correctly represent the layout of applications typically misrepresent the layout of abstractions, and vice versa<br>
 <br><img width="80%" src="https://github.com/mishaalkandapath/lambdaBERT/blob/main/lambdafigs/test_lev.png"><br>
40% of terms need ≤ 7 edits in order to be correct. ~29% need ≤ 1.
</div>
</p>
