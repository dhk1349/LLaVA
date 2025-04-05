# LLaVA Codebase Analysis

## Q1: What potential issues do you foresee when scaling the Llava codebase?
Scaling introduces significant complexity to the codebase. When scaling multiple aspects simultaneously (model size, data, and collaboration), the complexity increases exponentially. Additionally, training at a large scale (either model or data) inherently means fewer training iterations are possible, limiting our ability to validate and improve the model.

## Q2: How would you modify the code or architecture to address these issues?
The most effective solution lies in comprehensive information sharing and systematic tracking. Specifically:

- Implement robust tracking:
  - Hyperparameter and loss logging
  - Systematic checkpoint saving for experiment reproducibility
  - Data and code versioning

- Establish knowledge sharing practices:
  - Document rationale for each experiment version
  - Track expected outcomes and next steps
  - Share lessons learned from previous experiments

- Leverage automation:
  - Use GitHub Actions for code versioning and automated runs
  - Implement systematic running

## Q3: What modifications were necessary to utilize video and audio data?
The following modifications were implemented:

- Video support:
  - Adapted the architecture to handle multiple image frames
  - Integrated relevant components from LLaVA-NeXT
  - Added audio token support alongside image tokens

- Audio support:
  - Added wav2vec encoder and multimodal audio projector
  - Implemented audio token handling

- Testing:
  - Developed simplified test code for MacOS compatibility

## Files changed
```
.github/workflows/run_training.yml: example gitaction yaml file
llava/utils/logging_utils.py: logging example code
llava/model/multimodal_encoder/wav2vec_encoder.py: audio encoder
llava/model/multimodal_resampler: video frame handling from LLaVA-NeXT
llava/model/multimodal_projector/builder.py: audio projector
llava/model/llava_arch.py: video and audio processing from LLaVA-NeXT
llava/model/language_model/llava_llama.py: added audio input parameter
llava/train/train_dryrun.py: test run code
```

## Usage
```bash
python llava/train/train_dryrun.py --output_dir ./dryrun_outputs
```

check latest commit history to see the files changed.
