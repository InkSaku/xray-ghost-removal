# Project working rules

## Output retention

- Reuse stable output paths and overwrite superseded generated results after verifying their role.
- Before adding a result directory, inspect `outputs/` for an obsolete predecessor that can be replaced.
- Put one-off diagnostics, render checks, and test artifacts in an OS temporary directory or pytest `tmp_path`, never in the repository.
- Keep only the current machine result plus the minimum audit evidence needed to reproduce a frozen decision.
- Never delete raw data, the hashed SAM mask archive, frozen configs, or the final audit records they reference.
- Create a new versioned output only when the scientific definition or frozen evidence boundary changes; document why the old version must remain.
