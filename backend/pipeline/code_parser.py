"""tree-sitter based code parsing into graph-friendly structures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.pipeline.repo_loader import RepoFile

try:
    from tree_sitter import Language, Parser
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    Language = None
    Parser = None

_DOCSTRING_RE = re.compile(r'^\s*(?:"""|\'\'\'|"|\')(.*?)(?:"""|\'\'\'|"|\')', re.DOTALL)
_JS_COMMENT_RE = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)


@dataclass(slots=True)
class ParsedFunction:
    """A parsed function or method."""

    type: str
    name: str
    signature: str
    docstring: str | None
    body: str
    calls: list[str]
    line_start: int
    line_end: int
    file_path: str
    is_method: bool
    parent_class: str | None = None


@dataclass(slots=True)
class ParsedClass:
    """A parsed class."""

    type: str
    name: str
    docstring: str | None
    methods: list[str]
    bases: list[str]
    line_start: int
    line_end: int
    file_path: str


@dataclass(slots=True)
class ParsedImport:
    """A parsed import statement."""

    type: str
    source_file: str
    imported_module: str
    imported_names: list[str]


@dataclass(slots=True)
class ParsedFile:
    """All graph-relevant structures extracted from a source file."""

    file_path: str
    absolute_path: str
    language: str
    module_name: str
    raw_content: str
    functions: list[ParsedFunction] = field(default_factory=list)
    classes: list[ParsedClass] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)


class LanguageParser:
    """Base parser API for a single language."""

    language_name: str = ""

    def parse(self, file_path: str, raw_content: str) -> ParsedFile:
        raise NotImplementedError


class UnsupportedLanguageParser(LanguageParser):
    """Fallback parser for languages not yet implemented via tree-sitter."""

    def __init__(self, language_name: str) -> None:
        self.language_name = language_name

    def parse(self, file_path: str, raw_content: str) -> ParsedFile:
        return ParsedFile(
            file_path=file_path,
            absolute_path=file_path,
            language=self.language_name,
            module_name=_module_name(file_path),
            raw_content=raw_content,
        )


class TreeSitterLanguageParser(LanguageParser):
    """Shared helpers for tree-sitter powered parsers."""

    def __init__(self, language_name: str, language_binding: Any) -> None:
        if Language is None or Parser is None:
            raise ModuleNotFoundError(
                "tree-sitter is not installed. Run `pip install -r backend/requirements.txt`."
            )
        self.language_name = language_name
        self.language = _coerce_language(language_binding)
        self.parser = Parser()
        self.parser.language = self.language

    def _parse_tree(self, raw_content: str) -> tuple[Any, bytes]:
        source_bytes = raw_content.encode("utf-8")
        return self.parser.parse(source_bytes), source_bytes

    def _text(self, node: Any, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _walk(self, node: Any) -> list[Any]:
        stack = [node]
        nodes: list[Any] = []
        while stack:
            current = stack.pop()
            nodes.append(current)
            stack.extend(reversed(getattr(current, "children", [])))
        return nodes

    def _find_calls(self, node: Any, source_bytes: bytes) -> list[str]:
        calls: list[str] = []
        seen: set[str] = set()
        for child in self._walk(node):
            if child.type not in {"call", "call_expression"}:
                continue
            target = child.child_by_field_name("function")
            if target is None and getattr(child, "children", []):
                target = child.children[0]
            if target is None:
                continue
            call_name = self._normalize_call_name(target, source_bytes)
            if call_name and call_name not in seen:
                seen.add(call_name)
                calls.append(call_name)
        return calls

    def _normalize_call_name(self, node: Any, source_bytes: bytes) -> str:
        node_text = self._text(node, source_bytes).strip()
        return re.sub(r"\s+", "", node_text)

    def _docstring_from_body(self, body_text: str) -> str | None:
        match = _DOCSTRING_RE.search(body_text.strip())
        return match.group(1).strip() if match else None


class PythonParser(TreeSitterLanguageParser):
    """Python AST extraction using tree-sitter-python."""

    def __init__(self) -> None:
        import tree_sitter_python

        super().__init__("python", tree_sitter_python.language())

    def parse(self, file_path: str, raw_content: str) -> ParsedFile:
        tree, source_bytes = self._parse_tree(raw_content)
        parsed = ParsedFile(
            file_path=file_path,
            absolute_path=file_path,
            language="python",
            module_name=_module_name(file_path),
            raw_content=raw_content,
        )
        class_stack: list[str] = []
        method_map: dict[str, list[str]] = {}

        def visit(node: Any) -> None:
            if node.type == "class_definition":
                class_name = self._text(node.child_by_field_name("name"), source_bytes)
                body_node = node.child_by_field_name("body")
                body_text = self._text(body_node, source_bytes) if body_node else ""
                bases = self._python_bases(node, source_bytes)
                parsed_class = ParsedClass(
                    type="class",
                    name=class_name,
                    docstring=self._docstring_from_body(body_text),
                    methods=[],
                    bases=bases,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    file_path=file_path,
                )
                parsed.classes.append(parsed_class)
                method_map[class_name] = parsed_class.methods
                class_stack.append(class_name)
                for child in node.children:
                    visit(child)
                class_stack.pop()
                return

            if node.type == "function_definition":
                name = self._text(node.child_by_field_name("name"), source_bytes)
                body_node = node.child_by_field_name("body")
                body_text = self._text(node, source_bytes)
                signature = body_text.splitlines()[0].strip()
                function = ParsedFunction(
                    type="function",
                    name=name,
                    signature=signature,
                    docstring=self._docstring_from_body(self._text(body_node, source_bytes) if body_node else ""),
                    body=body_text,
                    calls=self._find_calls(node, source_bytes),
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    file_path=file_path,
                    is_method=bool(class_stack),
                    parent_class=class_stack[-1] if class_stack else None,
                )
                parsed.functions.append(function)
                if class_stack:
                    method_map[class_stack[-1]].append(name)
                return

            if node.type in {"import_statement", "import_from_statement"}:
                parsed_import = self._parse_import(node, source_bytes, file_path)
                if parsed_import:
                    parsed.imports.append(parsed_import)

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return parsed

    def _python_bases(self, node: Any, source_bytes: bytes) -> list[str]:
        bases: list[str] = []
        for child in node.children:
            if child.type == "argument_list":
                for grandchild in child.children:
                    if grandchild.type in {"identifier", "attribute"}:
                        bases.append(self._text(grandchild, source_bytes).strip())
        return bases

    def _parse_import(self, node: Any, source_bytes: bytes, file_path: str) -> ParsedImport | None:
        raw = self._text(node, source_bytes).strip()
        if node.type == "import_statement":
            modules = [part.strip() for part in raw.removeprefix("import ").split(",") if part.strip()]
            imported_module = modules[0].split(" as ")[0].strip() if modules else ""
            names = [module.split(" as ")[0].strip().split(".")[-1] for module in modules]
            if imported_module:
                return ParsedImport(
                    type="import",
                    source_file=file_path,
                    imported_module=imported_module,
                    imported_names=names,
                )
            return None

        match = re.match(r"from\s+([^\s]+)\s+import\s+(.+)", raw)
        if not match:
            return None
        imported_module = match.group(1).strip()
        imported_names = [
            name.strip().split(" as ")[0].strip()
            for name in match.group(2).split(",")
            if name.strip()
        ]
        return ParsedImport(
            type="import",
            source_file=file_path,
            imported_module=imported_module,
            imported_names=imported_names,
        )


class JavaScriptParser(TreeSitterLanguageParser):
    """JavaScript AST extraction using tree-sitter-javascript."""

    def __init__(self) -> None:
        import tree_sitter_javascript

        super().__init__("javascript", tree_sitter_javascript.language())

    def parse(self, file_path: str, raw_content: str) -> ParsedFile:
        tree, source_bytes = self._parse_tree(raw_content)
        parsed = ParsedFile(
            file_path=file_path,
            absolute_path=file_path,
            language="javascript",
            module_name=_module_name(file_path),
            raw_content=raw_content,
        )
        class_methods: dict[str, list[str]] = {}

        def visit(node: Any, active_class: str | None = None) -> None:
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                class_name = self._text(name_node, source_bytes) if name_node else "AnonymousClass"
                bases = []
                heritage = node.child_by_field_name("heritage")
                if heritage is not None:
                    bases.append(self._text(heritage, source_bytes).replace("extends", "").strip())
                parsed_class = ParsedClass(
                    type="class",
                    name=class_name,
                    docstring=_extract_js_docstring(raw_content, node.start_byte),
                    methods=[],
                    bases=bases,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    file_path=file_path,
                )
                parsed.classes.append(parsed_class)
                class_methods[class_name] = parsed_class.methods
                for child in node.children:
                    visit(child, class_name)
                return

            function = self._js_function_from_node(node, source_bytes, file_path, active_class)
            if function is not None:
                parsed.functions.append(function)
                if active_class:
                    class_methods[active_class].append(function.name)

            if node.type == "import_statement":
                parsed.imports.append(self._js_import(node, source_bytes, file_path))

            for child in node.children:
                visit(child, active_class)

        visit(tree.root_node)
        return parsed

    def _js_function_from_node(
        self, node: Any, source_bytes: bytes, file_path: str, active_class: str | None
    ) -> ParsedFunction | None:
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            body_node = node.child_by_field_name("body")
            body_text = self._text(node, source_bytes)
            return ParsedFunction(
                type="function",
                name=self._text(name_node, source_bytes) if name_node else "anonymous",
                signature=body_text.splitlines()[0].strip(),
                docstring=_extract_js_docstring(body_text, 0),
                body=body_text,
                calls=self._find_calls(node, source_bytes),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                file_path=file_path,
                is_method=False,
            )

        if node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            body_node = node.child_by_field_name("body") or node.child_by_field_name("value")
            body_text = self._text(node, source_bytes)
            return ParsedFunction(
                type="function",
                name=self._text(name_node, source_bytes) if name_node else "method",
                signature=body_text.splitlines()[0].strip(),
                docstring=_extract_js_docstring(body_text, 0),
                body=body_text if body_node is None else self._text(node, source_bytes),
                calls=self._find_calls(node, source_bytes),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                file_path=file_path,
                is_method=True,
                parent_class=active_class,
            )

        if node.type == "lexical_declaration":
            declarator = next((child for child in node.children if child.type == "variable_declarator"), None)
            if declarator is None:
                return None
            value = declarator.child_by_field_name("value")
            if value is None or value.type not in {"arrow_function", "function"}:
                return None
            name_node = declarator.child_by_field_name("name")
            body_text = self._text(node, source_bytes)
            return ParsedFunction(
                type="function",
                name=self._text(name_node, source_bytes) if name_node else "anonymous",
                signature=body_text.splitlines()[0].strip(),
                docstring=_extract_js_docstring(body_text, 0),
                body=body_text,
                calls=self._find_calls(value, source_bytes),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                file_path=file_path,
                is_method=False,
            )
        return None

    def _js_import(self, node: Any, source_bytes: bytes, file_path: str) -> ParsedImport:
        raw = self._text(node, source_bytes)
        module_match = re.search(r'from\s+[\'"]([^\'"]+)[\'"]', raw)
        imported_module = module_match.group(1) if module_match else ""
        names = re.findall(r"{([^}]+)}", raw)
        imported_names = []
        if names:
            imported_names = [name.strip() for name in names[0].split(",") if name.strip()]
        return ParsedImport(
            type="import",
            source_file=file_path,
            imported_module=imported_module,
            imported_names=imported_names,
        )


class TypeScriptParser(JavaScriptParser):
    """TypeScript AST extraction using tree-sitter-typescript."""

    def __init__(self) -> None:
        import tree_sitter_typescript

        language_binding = getattr(tree_sitter_typescript, "language_typescript", None)
        if not callable(language_binding):
            raise AttributeError("tree_sitter_typescript.language_typescript is unavailable.")
        TreeSitterLanguageParser.__init__(self, "typescript", language_binding())


def get_parser(language: str) -> LanguageParser:
    """Return the parser instance for a given language."""

    try:
        if language == "python":
            return PythonParser()
        if language == "javascript":
            return JavaScriptParser()
        if language == "typescript":
            return TypeScriptParser()
    except Exception:
        return UnsupportedLanguageParser(language)
    return UnsupportedLanguageParser(language)


def parse_repo_files(repo_files: list[RepoFile]) -> list[ParsedFile]:
    """Parse all repo files into graph-friendly structures."""

    parsed_files: list[ParsedFile] = []
    for repo_file in repo_files:
        parser = get_parser(repo_file.language)
        parsed = parser.parse(repo_file.file_path, repo_file.raw_content)
        parsed.absolute_path = repo_file.absolute_path
        parsed_files.append(parsed)
    return parsed_files


def _module_name(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix == ".py":
        return ".".join(path.with_suffix("").parts)
    return path.with_suffix("").as_posix().replace("/", ".")


def _extract_js_docstring(text: str, end_offset: int) -> str | None:
    matches = list(_JS_COMMENT_RE.finditer(text[:end_offset]))
    if not matches:
        return None
    return re.sub(r"^\s*\*\s?", "", matches[-1].group(1), flags=re.MULTILINE).strip() or None


def _coerce_language(language_binding: Any) -> Any:
    """Handle tree-sitter >=0.21 bindings across minor package variations."""

    try:
        return Language(language_binding)
    except Exception:
        return language_binding
