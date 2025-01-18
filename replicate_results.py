from refined.evaluation.evaluation import eval_all
from refined.inference.processor import Refined

finetuned_path = "/root/autodl-tmp/ReFinED/example_scripts/fine_tuned_models/1733808457/f1_0.9180/"
refined = Refined.from_pretrained(model_name=finetuned_path,
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)
print('EL results (with model fine-tuned on AIDA)')
eval_all(refined=refined, el=True, filter_nil_spans=False)

# refined = Refined.from_pretrained(model_name='wikipedia_model',
#                                   entity_set='wikipedia',
#                                   use_precomputed_descriptions=True)
print('ED results (with model fine-tuned on AIDA)')
eval_all(refined=refined, el=False)
