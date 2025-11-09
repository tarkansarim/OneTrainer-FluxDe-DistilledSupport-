# SRPO Dataset Handoff → Full Integration

1. Export OneTrainer Concepts
   - Generate an SRPO-compatible `videos2caption2.json` under the selected SRPO working directory using the active concept file.

2. Inject SRPO Script Overrides
   - Extend the launcher so it automatically adds dataset/reward paths (unless overridden) when launching the SRPO script.

3. UI & Automation Enhancements
   - Surface SRPO-specific training parameters in the Training tab and wire them into the launcher.
   - Automatically ensure required SRPO assets (Flux base model, HPS checkpoints, optional PickScore) are present or downloaded.

4. Windows Compatibility
   - Ensure the SRPO launcher can execute on Windows without requiring a user-supplied bash by adding PowerShell-native handling or direct Python entrypoint.

5. Progress Feedback
   - Improve SRPO dataset preparation logging/status updates so the user sees batch progress instead of long silent periods.
