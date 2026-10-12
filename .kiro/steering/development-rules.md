---
inclusion: always
---
<!------------------------------------------------------------------------------------
   Add rules to this file or a short description and have Kiro refine them for you.
   
   Learn about inclusion modes: https://kiro.dev/docs/steering/#inclusion-modes
-------------------------------------------------------------------------------------> 
# PM RAID Agent — Development Rules

## 1. Source of Truth
- Read the current `requirements.md`, `design.md`, and `tasks.md` directly from disk before implementing a task.
- Follow the specifications in this order: Requirements → Design → Tasks.
- Do not rely on previous chat context when it conflicts with the current files.

## 2. Scope Control
- Implement only the task explicitly requested by the user.
- Do not introduce features, providers, dependencies, or infrastructure outside the approved design.
- Do not modify Requirements, Design, or Tasks without explicit user authorization.

## 3. Change Management
- Preserve existing files and user modifications.
- Never delete, overwrite, regenerate, or recreate existing work without explicit authorization.
- Before modifying an existing file, inspect its current content.
- If there are pending review changes or conflicting file states, stop and ask the user before proceeding.

## 4. Implementation and Validation
- Keep implementation simple and appropriate for the MVP.
- Run relevant tests and report actual results.
- Never claim tests passed unless they were executed successfully.
- Report files changed, completed work, and remaining issues.
- Stop after the assigned task and wait for further instructions.

## 5. Tool Execution
- Environment: Windows PowerShell 5.1.
- Prefer native IDE tools for file operations.
- Use only PowerShell 5.1-compatible commands.
- Do not use Bash-specific syntax or options.
- If a tool operation fails twice, stop and report the blocker.