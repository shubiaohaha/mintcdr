from __future__ import annotations

from pathlib import Path


class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(f.read())

    def render(self, **kwargs: str) -> str:
        return self.template.format(**{key: value or "" for key, value in kwargs.items()})


def build_item_utility_prompt(
    item_description: str,
    item_attributes: str,
    source_domain_description: str,
    target_domain_description: str,
    template_path: str | Path = "prompts/item_utility_prompt.txt",
) -> str:
    template = PromptTemplate.from_file(template_path)
    return template.render(
        item_description=item_description,
        item_attributes=item_attributes,
        source_domain_description=source_domain_description,
        target_domain_description=target_domain_description,
    )

