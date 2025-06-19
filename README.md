# EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering

## 📖 Paper

[More Information Needed]

## 🔎 About EsBBQ and CaBBQ

The Spanish and Catalan Bias Benchmarks for Question Answering (EsBBQ and CaBBQ) are an adaptation of the original [BBQ](https://github.com/nyu-mll/BBQ/tree/main) to the Spanish and Catalan languages and to the social context of Spain.
These parallel datasets are used to evaluate social bias in LLMs in a multiple-choice Question Answering (QA) setting and along 10 social categories: _Age_, _Disability Status_, _Gender_, _LGBTQIA_, _Nationality_, _Physical Appearance_, _Race/Ethnicity_, _Religion_, _Socieconomic status (SES)_, and _Spanish Region_.

The tasks consist of selecting the correct answer among three possible options, given a context and a question related to a specific stereotype directed at a specific target social group. 
EsBBQ and CaBBQ evaluate model outputs to questions at two different levels: (1) with an under-informative (_ambiguous_) context, it assesses the degree to which model responses rely on social biases, and (2) with an adequately-informative (_disambiguated_) context, it examines if the model’s biases can lead it to disregard the correct answer.

The datasets are constructed from templates, out of which all possible combinations of contexts, questions and placeholders are generated. 

### Statistics

| **Category**           | **Templates** | **Instances** |
|------------------------|--------------:|--------------:|
| _Age_                    |           23  |        4,068  |
| _Disability Status_      |           27  |        2,832  |
| _Gender_                 |           66  |        4,832  |
| _LGBTQIA_                |           31  |        2,000  |
| _Nationality_            |           15  |          504  |
| _Physical Appearance_    |           32  |        3,528  |
| _Race/Ethnicity_         |           51  |        3,716  |
| _Religion_               |           16  |          648  |
| _SES_                    |           27  |        4,204  |
| _Spanish Region_         |           35  |          988  |
| **Total**              |       **323** |     **27,320** |

## 📁 Repository Structure

- `templates`: folder containing the `.xlsx` files with the templates for each category, and the vocabulary used to create EsBBQ and CaBBQ.
- `generate_instances.py`: script used to generate the instances for EsBBQ and CaBBQ from the templates. Adapted from the [script used for BBQ](https://github.com/nyu-mll/BBQ/blob/main/generate_from_template_all_categories.py).
- `utils.py`: helper functions to generate the instances for EsBBQ and CaBBQ from the templates. Adapted from the [script used for BBQ](https://github.com/nyu-mll/BBQ/blob/main/utils.py). 
- `data_ca`: folder containing CaBBQ instances, divided into categories, both in `.jsonl` and `.csv`.
- `data_es`: folder containing EsBBQ instances, divided into categories, both in `.jsonl` and `.csv`.
- `bias_score.py`: functions to calculate the accuracy and bias scores.
- `instance_language-revision.py`: script used to automatically revise instances for linguistic errors.

## ⚖️ Ethical Considerations

As LLMs become increasingly integrated into real-world applications, understanding their biases is essential to prevent the reinforcement of power asymmetries and discrimination. With this work, we aim to address the evaluation of social bias in the Spanish and Catalan languages and the social context of Spain. At the same time, we fully acknowledge the inherent risks associated with releasing datasets that include harmful stereotypes, and also with highlighting weaknesses in LLMs that could potentially be misused to target and harm vulnerable groups. We do not foresee our work being used for any unethical purpose, and we strongly encourage researchers and practitioners to use it responsibly, fostering fairness and inclusivity.

## 📜 License

This project is distributed under Apache-2.0 license.

## 📫 Contact

Language Technologies Unit (langtech@bsc.es) at the Barcelona Supercomputing Center (BSC). 

## 🥇 Acknowledgements

This work has been promoted and financed by the Generalitat de Catalunya through the Aina project, and by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project Desarrollo Modelos ALIA.

## 🖇️ Citation

[More information needed]
