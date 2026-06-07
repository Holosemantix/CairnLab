# AI Project Attribute Extraction Prompt

Given a project intake YAML and any repository notes, infer its facets. Use the taxonomy as guidance, but keep an open vocabulary.

Do not force a single domain or layer. Output all applicable attributes.

Questions:

1. What research fields does the project target?
2. What does it start from: topic, paper, repo, dataset, experiment plan, review, or other?
3. What does it output?
4. Which workflow scopes does it cover?
5. How deep is execution: text-only, code generation, smoke test, full training, fresh-container reproduction, wetlab loop, benchmark grading?
6. What verification model is used?
7. What accountability features exist?
8. What agent topology is used?
9. What integration style does it use?
10. What maturity signals and risks are visible?
11. Which existing projects are most similar, and why?
12. Does it introduce a new attribute not in the taxonomy?

Return YAML only.
