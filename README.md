# PM RAID Agent

**LLM-Assisted Project Document Analysis | LangGraph + Groq + Python**

## Overview

PM RAID Agent is a hands-on AI engineering project exploring how Large Language Models (LLMs) can support software project delivery by analyzing project documents and identifying potential Risks, Assumptions, Issues, and Dependencies (RAID).

The project focuses on combining LLM-based reasoning with a structured workflow and human review.

## Problem

Important delivery risks and dependencies are often scattered across project status reports, meeting minutes, and planning documents.

Manually reviewing these documents can be repetitive and time-consuming.

## How It Works

```text
Project Documents (PDF / DOCX / XLSX)
                |
                v
         Text Extraction
                |
                v
        Groq LLM Analysis
                |
                v
       Structured RAID Findings
       - Type and description
       - Supporting evidence
       - Recommended action
                |
                v
          PM Review
        Approve / Reject
                |
                v
       Excel RAID Register
       (Approved items only)
```

## Technology Stack

- Python — application development
- LangGraph — workflow orchestration
- Groq LLM — document analysis and RAID identification
- Streamlit — user interface
- Pytest — automated testing

## MVP Capabilities

- Process PDF, DOCX, and XLSX documents.
- Use an LLM to identify potential RAID items.
- Generate structured findings with supporting evidence and recommended actions.
- Allow a Project Manager to review and approve or reject findings.
- Export approved findings to Excel.

## Current Limitations

This is an experimental MVP, not a production-ready continuous monitoring system.

- Documents are uploaded manually.
- Cross-document AI synthesis is not fully operational.
- Automatic folder monitoring and incremental change detection are not implemented.
- Findings require human validation.

## Key Learning

Building an AI application is not automatically better than using a general-purpose LLM directly.

The longer-term opportunity is continuous RAID monitoring: detecting changes in project documents, comparing findings against an existing RAID register, and proposing updates for PM approval.

## Project Status

**Experimental MVP — completed initial user acceptance testing.**

This repository demonstrates practical experimentation with LLM integration, LangGraph workflow orchestration, and AI-assisted project delivery.