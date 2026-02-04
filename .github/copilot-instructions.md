# Refactoring Instructions: WSI-MIL Research Framework

## 1. Objective
Refactor the current codebase into a highly modular, configuration-driven MIL framework that allows rapid testing of new research ideas (e.g., multi-modal fusion, custom attention, voting strategies).

## 2. Structural Requirements (The "Golden Rules")
- **Interface Uniformity**: Every model/module must use `forward(self, data_dict: Dict) -> Dict`. 
    - Input `data_dict` contains at least `{'features': Tensor, 'coords': Tensor}`.
    - Output `dict` contains at least `{'logits': Tensor}`.
- **Factory Pattern**: Centralize object creation. Use a `Registry` or `Factory` class to instantiate models and datasets based on string names from config.
- **Configuration over Code**: Hard-coded values are forbidden. Extract all parameters to a unified `config` object/dict.

## 3. Data Refactoring (WSI/H5)
- Standardize the `WSIDataset` to handle `.h5` files.
- The dataset must be "agnostic" to the experiment—it just provides the features. Any feature manipulation (like IHC+HE concatenation) should happen in a separate `Transform` or a `Fusion` module within `models/`.

## 4. Model Refactoring
- **Inheritance**: All MIL models must inherit from a common `BaseMIL` class. Remove coupling with MIL-Lab.
- **Componentization**: Split models into `FeatureEncoder`, `Aggregator`, and `Classifier`.
- **Shape Documentation**: Every `forward` method MUST have shape comments. 
    - E.g., `# features: (B, N, C) where N is instance count.`

## 5. Refactoring Workflow (How to act)
- When I ask to refactor a file, first analyze the existing logic.
- Propose the new structure that follows the above rules before writing code.
- Ensure backward compatibility or provide a clear migration path for existing `.h5` data.