# How Eloundou et al. Constructed Occupation-Level AI Exposure Measures

## Overview

This document explains how Eloundou et al. (2024) transformed their task-level AI exposure ratings into occupation-level measures that were subsequently used by Brynjolfsson et al. (2025) and other researchers. The process involves multiple stages of data processing, weighting, and aggregation.

## Data Sources

### Primary Inputs
1. **O*NET Database (v27.2)**: Contains occupation and task information
2. **Task Exposure Labels**: Human and GPT-4 ratings for each task using the E0/E1/E2/E3 rubric
3. **Task Ratings**: Importance and relevance weights from O*NET
4. **BLS Employment Data**: Employment and wage statistics by occupation

## Step-by-Step Construction Process

### Step 1: Categorical to Numerical Conversion

The raw exposure categories (E0, E1, E2, E3) are converted to numerical scores using three different mapping schemes:

```python
alpha_score_map = {"E0": 0.0, "E1": 1.0, "E2": 0.0, "E3": 0.0}
beta_score_map = {"E0": 0.0, "E1": 1.0, "E2": 0.5, "E3": 0.5}
gamma_score_map = {"E0": 0.0, "E1": 1.0, "E2": 1.0, "E3": 1.0}
```

**Key Variables Created:**
- `gpt4_alpha`: Only E1 tasks get score of 1.0 (direct LLM exposure only)
- `gpt4_beta`: E1 = 1.0, E2/E3 = 0.5 (partial credit for software-enhanced exposure) ⭐ **This is what Brynjolfsson et al. use**
- `gpt4_gamma`: E1/E2/E3 all get 1.0 (any exposure counts fully)
- `human_alpha`, `human_beta`, `human_gamma`: Same mappings for human ratings

### Step 2: Task Weighting Schemes

Four different weighting approaches are applied:

1. **Equal Weight**: All tasks weighted equally
2. **Core Weight**: Core tasks get 2x weight of supplemental tasks  
3. **Importance Weight**: Based on O*NET importance ratings
4. **Relevance Weight**: Based on O*NET relevance ratings

```python
# Core weighting example
df_tasks["coreweight"] = df_tasks["Task Type"].map(
    {"Core": 2, "Supplemental": 1, np.nan: 1}
)
```

### Step 3: Occupation-Level Aggregation

Tasks are aggregated to occupation level using weighted averages:

```python
def weighted_mean(df, groupfields, aggfields, weightfield):
    df2 = df[aggfields].multiply(df[weightfield], axis="index")
    aa = df[[weightfield] + groupfields]
    df3 = df2.join(aa)
    dfg = df3[groupfields + aggfields].groupby(groupfields).sum().reset_index()
    return dfg
```

**Resulting Variables in `occ_level`:**
- `gpt4_alpha`: Occupation-level alpha scores
- `gpt4_beta`: Occupation-level beta scores ⭐ **Used by Brynjolfsson et al.**
- `gpt4_gamma`: Occupation-level gamma scores
- `human_alpha`, `human_beta`, `human_gamma`: Human equivalents
- `automation`: Automation scores (T0-T4 mapped to 0.0-1.0)

### Step 4: Integration with Labor Market Data

The occupation-level exposure measures are merged with:

1. **BLS Employment Data**: Total employment, wages by occupation
2. **Demographics**: Worker characteristics by occupation
3. **Skills Data**: O*NET skill requirements
4. **Work Context**: Physical and social work environment

```python
# Example merge with BLS wage/employment data
occ_level["OCC_CODE"] = occ_level["O*NET-SOC Code"].str.slice(start=0, stop=7)
occ_lvl = pd.merge(
    occ_level,
    raw_bls[["OCC_CODE", "TOT_EMP", "H_MEAN", "A_MEAN", "H_MEDIAN", "A_MEDIAN"]],
    how="left",
    on="OCC_CODE",
)
```

## Key Output Variables in `occ_level`

### Exposure Measures (What Brynjolfsson et al. Use)
- **`gpt4_beta`**: GPT-4 β exposure (E1 + 0.5×E2 + 0.5×E3)
- **`human_beta`**: Human β exposure (same formula, human ratings)
- **`gpt4_alpha`**: Direct exposure only (E1 only)
- **`gpt4_gamma`**: Full exposure (E1 + E2 + E3)
- **`automation`**: Automation potential scores

### Labor Market Variables
- **`TOT_EMP`**: Total employment in occupation
- **`A_MEAN`**: Annual mean wage
- **`H_MEAN`**: Hourly mean wage
- **`log_A_mean`**: Log of annual mean wage
- **`log_totemp`**: Log of total employment

### Occupation Identifiers
- **`O*NET-SOC Code`**: Full 8-digit occupation code
- **`OCC_CODE`**: 7-digit BLS occupation code
- **`Title`**: Occupation title
- **`simpleOcc`**: Simplified occupation code

## The β Measure: E1 + 0.5×E2 + 0.5×E3

The β score represents **partial implementation exposure**:

- **E1 tasks (score = 1.0)**: Can be improved by direct LLM access alone
- **E2 tasks (score = 0.5)**: Require additional software beyond LLMs
- **E3 tasks (score = 0.5)**: Require image processing capabilities

**Economic Interpretation**: This measure assumes that complementary technologies (software, image processing) will be partially but not fully developed, making E2/E3 tasks only 50% as exposed as E1 tasks.

## Industry-Level Analysis

The code also creates industry-level exposure measures:

1. **Employment-weighted aggregation**: Uses actual employment counts to weight occupations within industries
2. **NAICS classification**: 2-digit, 3-digit, and 4-digit industry codes
3. **Productivity data integration**: Merges with BEA/BLS productivity statistics

## Validation and Robustness

The methodology includes several validation steps:

1. **Human-GPT-4 agreement**: Correlation analysis between human and model ratings
2. **Alternative weighting schemes**: Sensitivity analysis across different task weights  
3. **Cross-validation**: Comparison with existing automation measures (Frey & Osborne, Webb, etc.)

## Usage in Subsequent Research

**Brynjolfsson et al. (2025)** specifically use:
- **`gpt4_beta`** as their main exposure measure
- Create exposure quintiles for regression analysis
- Use these measures to predict employment changes by age group

**Other researchers** have used:
- **`gpt4_alpha`**, **`gpt4_gamma`** for robustness checks
- **`human_beta`** for comparison with model-based measures
- **`automation`** scores for analyzing substitution vs. complementarity

## Technical Notes

- **Missing data handling**: Uses list-wise deletion for missing O*NET ratings
- **Occupation matching**: Requires exact matches between O*NET and BLS codes
- **Weighting sensitivity**: Results shown to be robust across different weighting schemes
- **Sample coverage**: Covers ~800-900 occupations depending on data availability

This occupation-level dataset (`occ_level`) forms the foundation for most subsequent econometric analyses of AI's labor market impacts, including the employment effects documented in Brynjolfsson et al. (2025).

## References

``` Bibligraphy
@article{eloundou_gpts_2024,
	title = {{GPTs} are {GPTs}: {Labor} market impact potential of {LLMs}},
	volume = {384},
	shorttitle = {{GPTs} are {GPTs}},
	url = {https://www.science.org/doi/full/10.1126/science.adj0998},
	doi = {10.1126/science.adj0998},
	language = {en-US},
	number = {6702},
	urldate = {2025-08-19},
	journal = {Science},
	author = {Eloundou, Tyna and Manning, Sam and Mishkin, Pamela and Rock, Daniel},
	month = jun,
	year = {2024},
	pages = {1306--1308},
}

@article{brynjolfsson_canaries_2025,
	title = {Canaries in the {Coal} {Mine}? {Six} {Facts} about the {Recent} {Employment} {Effects} of {Artificial} {Intelligence}},
	language = {en},
	author = {Brynjolfsson, Erik and Chandar, Bharat and Chen, Ruyu},
	month = aug,
	year = {2025},
}
```