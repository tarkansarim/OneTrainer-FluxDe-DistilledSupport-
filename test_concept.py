#!/usr/bin/env python3

import json
import sys
sys.path.insert(0, '.')

from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress
from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin

# Load config
config_path = "workspace/animereal2/config/2025-11-10_19-03-29.json"
with open(config_path, 'r') as f:
    config_dict = json.load(f)

config = TrainConfig.default_values()
config = config.from_dict(config_dict)

# Create a mock data loader mixin
class MockDataLoader(DataLoaderMgdsMixin):
    def __init__(self):
        self.train_device = 'cpu'

# Test the concept loading
mock_loader = MockDataLoader()

concepts = config.concepts
if concepts is None:
    with open(config.concept_file_name, 'r') as f:
        raw_concepts = json.load(f)
        print(f"Loaded {len(raw_concepts)} raw concepts from file")
        concepts = [TrainConfig.ConceptConfig.default_values().from_dict(c) for c in raw_concepts]

print(f"After loading: {len(concepts)} concepts")

# Filter for validation (is_validation=False)
from modules.util.config.TrainConfig import ConceptType
concepts = [concept for concept in concepts if (ConceptType(concept.type) == ConceptType.VALIDATION) == False]
print(f"After validation filter: {len(concepts)} concepts")

# Convert to dict
concepts_dict = [c.to_dict() for c in concepts]
print(f"After to_dict: {len(concepts_dict)} concepts")

# Test ConceptPipelineModule
from mgds.ConceptPipelineModule import ConceptPipelineModule
concept_module = ConceptPipelineModule(concepts_dict)
print(f"ConceptPipelineModule length: {concept_module.length()}")
