import re  # noqa
from typing import List, Dict, Optional
import toml
import os
from api.sillicon_rerank import Rerank
from api.openai_call import OpenToolCall



class MessageEval:
    "session内评分,支持动态的多rules价值函数"
    def __init__(self,messages: list):
        self.messages = messages    
        self._score = 0
        self._score_party_count = 0
        self.llm_messages = self.get_messages(["assistant","tool"])

    def __len__(self):
        return len(self.messages)

    def get_messages(self,roles:List[str]) -> list:
        return [message for message in self.messages if message["role"] in roles]
    
    @property
    def score(self):
        return round(self._score / self._score_party_count,1) * 10
    
    def compute(self):
        self.has_success_rag()
        self.has_success_tools()
        self.has_success_plan()
        self.has_success_sql()
        self.has_success_terminal()
        self.has_success_html()
        self.has_success_python()
        return self.score

    def has_success_rag(self) -> bool:
        "具有 rag 调用, 监测通用 rag 调用(function name: rag_query)"
        has_rag = any([msg['message']['name'] == "rag_query" for msg in self.messages if msg['role'] == "tool"])
        self._score += 1 if has_rag else 0
        self._score_party_count += 1
        return has_rag

    def has_success_tools(self) -> bool:
        "具有工具调用, 监测通用工具调用(包含rag的工具调用)"
        has_tool = any([msg['message']['name'] != "rag_query" for msg in self.messages if msg['role'] == "tool"])
        self._score += 1 if has_tool else 0
        self._score_party_count += 1
        return has_tool
    
    def has_web_search_tools(self) -> bool:
        "具有工具调用, 监测通用工具调用(包含rag的工具调用)"
        has_tool = any([msg['message']['name'] != "rag_query" for msg in self.messages if msg['role'] == "tool"])
        self._score += 1 if has_tool else 0
        self._score_party_count += 1
        return has_tool

    def has_success_plan(self):
        """具有计划调用, 监测通用计划调用(function name: planning)"""
        has_plan = any([msg['message']['name'] == "planning" for msg in self.messages if msg['role'] == "tool"])
        self._score += 1 if has_plan else 0
        self._score_party_count += 1
        return has_plan

    def has_success_sql(self):
        """具有 sql 调用, 监测通用 sql"""
        tmp_sql_tool_name = "mcp-clickhouse-global-access-run_select_query"
        has_sql = any([msg['message']['name'] == tmp_sql_tool_name for msg in self.messages if msg['role'] == "tool"])
        self._score += 1 if has_sql else 0
        self._score_party_count += 1
        return has_sql

    def has_success_terminal(self):
        """具有 终端 调用, 监测通用终端调用(function name: terminal_query)"""
        has_terminal = any([msg['message']['name'] == "terminate" for msg in self.messages if msg['role'] == "tool"])
        self._score += 1 if has_terminal else 0
        self._score_party_count += 1
        return has_terminal

    def has_success_html(self):
        """具有 html 调用, 监测通用 html 调用(function name: html_query)"""
        p = re.compile(r".*```[ \t]*html.+```.*",re.DOTALL)
        has_html = any([p.match(msg['message']['content']) for msg in self.messages if msg['role'] == "assistant"])
        self._score += 1 if has_html else 0
        self._score_party_count += 1
        return has_html
    
    def has_success_python(self):
        """具有 python 调用, 监测通用 python 调用(function name: python_query)"""
        p = re.compile(r".*```[ \t]*python.+```.*",re.DOTALL)
        has_html = any([p.match(msg['message']['content']) for msg in self.messages if msg['role'] == "assistant"])
        self._score += 1 if has_html else 0
        self._score_party_count += 1
        return has_html
    
    def eval_with_llm(self):
        """
        大模型的评分
        上文中对话内容:
        1. (is_full_content)是否详尽完整? 
        2. (is_virtual_assumption)是否存在虚拟的假设? 
        3. (is_pending_content)是否有留待完成的内容? 
        4. (is_source_from_tools)如果有数字,数学,统计,计算,金融,数据库等信息, 其来源是否都是从工具中调取? 
        5. (is_meet_requirements)总体的回答是否符合我提出的要求? 
        """
        ...
        

class BaseRule:
    def __init__(
        self,
        action: str,
        role: List[str],
        pattern: Optional[str] = None,
        text: Optional[str] = None,
        prompt: List = [],
        values: List[int] = [],
    ):
        self.action = action
        self.roles = role
        self.text = text
        self.values: List[int] = values
        self.pattern = re.compile(pattern, re.DOTALL) if pattern else None
        self.prompt_tools = self.build_function(prompt) if prompt else None
        self.score_patry_count = 1 if not prompt else len(prompt)
        self.need_llm = True if self.prompt_tools else False

    def content_check(self, content: str) -> bool:
        if self.text:
            return self.text == content
        if self.pattern is None:
            return False
        matched = True if self.pattern.match(content) else False
        return matched

    def build_function(self, prompt: List) -> dict:
        required = [item["name"] for item in prompt]
        description = "以上助手的回答, 是否满足如下要求?\n" + "\n".join(
            [item["content"] for item in prompt]
        )
        properties = {
            item["name"]: {
                "type": "boolean",
                "description": item["content"],
            }
            for item in prompt
        }
        function = {
            "type": "function",
            "function": {
                "name": "review_content",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return function

    def regex_rule(self, content: str) -> bool:
        return self.content_check(content)

    def contains_rule(self, content: str) -> bool:
        if self.text is None:
            return True
            
        return self.text in content

    def tools_rule(self, function_name: str) -> bool:
        return self.content_check(function_name)

    def __call__(self, messages: List[Dict],toolcall_model: Optional[OpenToolCall] = None) -> int:
        checked = False
        if self.action == "regex":
            checked = any(
                [
                    self.regex_rule(msg["message"]["content"])
                    for msg in messages
                    if msg["role"] in self.roles
                ]
            )
        elif self.action == "contains":
            checked = any(
                [
                    self.contains_rule(msg["message"]["content"])
                    for msg in messages
                    if msg["role"] in self.roles
                ]
            )
        elif self.action == "function":
            checked = any(
                [
                    self.tools_rule(msg["message"]["name"])
                    for msg in messages
                    if msg["role"] in self.roles
                ]
            )
        elif self.action == "llm":
            if toolcall_model is None:
                raise ValueError("llm rule requires a toolcall model")
            score = toolcall_model.ensure_tool_call(messages,self.prompt_tools) # noqa  # pyright: ignore[reportArgumentType]
            return score
        elif self.action == "steps_count" and len(self.values) >= 2:
            resp_len = sum([1 for msg in messages if msg["role"] in self.roles])
            checked = self.values[0] <= resp_len <= self.values[1]
        return 1 if checked else 0


class LoadRules:
    def __init__(self, dirpath: str):
        files = [
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.lower().endswith(".toml")
        ]
        print(f"load rules: {len(files)}")
        # name: string, description: string, rules: list
        self.groups = [toml.load(f) for f in files]
        self.rules = [
            BaseRule(**rule) for group in self.groups for rule in group["rules"]
        ]
        self._score_party_count = sum([rule.score_patry_count for rule in self.rules])
        self._rerank_model: Optional[Rerank] = None
        self._toolcall_model: Optional[OpenToolCall] = None
        self._system_prompt_cache:dict = {}

    def __len__(self):
        return len(self.groups)
    
    @property
    def rerank_model(self) -> Rerank:
        if self._rerank_model is None:
            self._rerank_model = Rerank()
        return self._rerank_model
    
    @property
    def toolcall_model(self) -> OpenToolCall:
        if self._toolcall_model is None:
            self._toolcall_model = OpenToolCall()
        return self._toolcall_model

    def score_one(self, messages: List[Dict]) -> float:
        "通用规则打分"
        score = sum([rule(messages) if not rule.need_llm else rule(messages,self.toolcall_model) for rule in self.rules])
        return round(score / self._score_party_count, 1) * 10

    def rank_grouped(self, messages: List[Dict]) -> int:
        "根据agent的系统提示词进行分类,每个分类使用不同的评分规则"
        system_contents = [m['message']['content'] for m in messages if m['role'] == 'system']
        content_key = "\n".join(system_contents)
        if content_key in self._system_prompt_cache:
            return self._system_prompt_cache[content_key]
        
        documents = [g['description'] for g in self.groups]
        group_index = self.rerank_model.query(content_key, documents)
        self._system_prompt_cache[content_key] = group_index
        return group_index
    
    def score_grouped(self,rule_index:int, messages: List[Dict]) -> float:
        "智能搜索打分"
        rule = self.rules[rule_index]
        score = rule(messages) if not rule.need_llm else rule(messages,self.toolcall_model)
        return round(score / self._score_party_count, 1) * 10

    def score(self, messages: List[Dict]) -> float:
        if len(self.groups) > 1:
            rule_index = self.rank_grouped(messages)
            return self.score_grouped(rule_index, messages)
        else:
            return self.score_one(messages)


if __name__ == "__main__":
    rules = LoadRules("conf")
    print(rules.groups)
