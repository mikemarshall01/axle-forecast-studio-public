# Task-based review policy

This policy applies to the Python Forecast Studio. A review unit is one coherent outcome with its implementation, relevant tests and necessary explanation. It is not a file count or an entire implementation wave. The lead releases a task only when its prerequisites and exclusive file ownership are clear.

## Risk and checks

| Risk | Examples | Before editing | Before integration |
| --- | --- | --- | --- |
| Routine | Wording, formatting, isolated static UI or fixture maintenance | Short intent and owned paths | Relevant mechanical checks and a scoped review when substantive |
| Normal | One bounded model helper, configuration parser, UI interaction or export | Outcome, acceptance criteria and applicable contract | Focused tests, self-diff check and one independent fresh-context review |
| High | Physical stock-flow, random pairing, planner cut-offs, six-route finance, source claims, result schemas or viewer isolation | Lead-accepted shared interface and tiny executable fixture; early specialist review when downstream tasks depend on it | Focused invariant/path tests and one strong independent review; add a second specialist only for a distinct material risk |
| Milestone/release | Integrated usable app or release candidate | Candidate revision and user journey | Combined checks and independent `docs/VERIFY_APP.md` evidence |

The writer runs fast checks while working. Ruff, pytest, schema checks and other mechanical tools own deterministic rules; an LLM review focuses on behaviour and interactions. Run affected tests for the task, then broader combined tests after integration. Do not demand a full agent review after each file or a full Monte Carlo run on ordinary edits. A full simulation in the app starts only from an explicit valid Run click.

For an affected model slice, the reviewer checks the relevant units and interval duration; event ordering and battery-energy conservation; physical 0–100% SoC, reserve and charger limits; one dispatchable route per EV-slot; stable population and paired exogenous worlds; decision-time information cut-offs; and world-first aggregation before quantiles. It also checks that observed data, assumptions, synthetic history and illustrative commercial outputs are labelled correctly. Use tiny hand-checkable fixtures for the affected rules rather than demanding every check from every unrelated slice.

## Writer handoff and frozen target

The writer inspects its own complete diff and Git status. In an assigned isolated worktree it may create a local commit containing only owned task files; it must never include pre-existing or unrelated changes. A frozen complete diff is also acceptable when a local commit is impractical. The handoff states:

- task outcome and acceptance criteria;
- base revision and exact review commit, or the frozen diff location and hash;
- owned files changed and why;
- commands actually run, results and checks not run;
- interface assumptions, blockers and remaining risks.

The reviewer receives the requirements, accepted decisions, complete task change, relevant surrounding code and test evidence. It does not inherit the writer's reasoning, edit the work, or review its own implementation. Do not point a reviewer at a branch the writer is still changing. The writer may prepare a different independent task in another worktree while review runs.

## Review and repair

One fresh independent reviewer is the default. Add a second specialist only when a distinct material risk needs it. The reviewer reports `PASS`, `CHANGES_REQUIRED` or `BLOCKED`, with findings labelled:

- `BLOCKING`: a plausible correctness or regression failure, unsafe behaviour, unsupported claim, broken contract or unmet acceptance criterion. Give location, failure scenario, impact and evidence.
- `QUESTION`: an unresolved meaning or decision that affects correctness. State the exact decision needed.
- `OPTIONAL`: style or improvement that does not violate an agreed rule. This does not block integration.

Do not invent a minimum number of findings. The writer or designated editor fixes blocking findings; the reviewer checks the updated snapshot and affected interactions. After two unsuccessful repair/review cycles, the lead diagnoses, decides or rescopes. No defect is waived automatically. A mechanical-only correction can receive a focused check rather than a repeat of an unrelated broad review.

## Integration and completion

The lead accepts shared contracts and integrates reviewed tasks one at a time. Review conflict resolutions or changed interactions, run affected checks on combined code and record the integrated revision. A worker's `finished` status is not approval. A task is `DONE` only after review and integrated validation; an app milestone is complete only after independent Python/Streamlit verification. Pushing, deployment and remote changes remain separate approvals.

Use factual status (`WRITING`, `READY_FOR_REVIEW`, `CHANGES_REQUIRED`, `APPROVED`, `INTEGRATED` or `BLOCKED`), a named owner, blocker and next action. Percentage complete and number of occupied agents are not release criteria.
