import pandas as pd
from api.pg_data import PostgreSQLData
import re
from typing import List

class EvalOrRL:
    def __init__(self,pg: PostgreSQLData):
        self.pg = pg

    def load_v2_agents(self):
        sql = 'select id as "agentId","llmId",name as "agentName","updatedAt",responsibilities from "Agents";'  
        agent_df = self.pg.read_sql_to_df(sql)
        # models
        sql = 'select id as "llmId",model,type,"maxTokens",temperature from "LlmConfigs";'
        model_df = self.pg.read_sql_to_df(sql)
        agent_df = agent_df.merge(model_df, left_on="llmId",right_on="llmId", how="left")
        agent_df = agent_df.drop(columns=["llmId"])
        # agentId,agentName,updatedAt,responsibilities,model
        return agent_df
    
    def load_v2_session_content(self):
        sql = 'select * from "V2AgentMessages" order by "created_at"'
        session_df = self.pg.read_sql_to_df(sql)
        # group by session_id order by created_at
        return session_df

    def load_v1_agent(self):
        sql = 'select id,"outId","llmId","updatedAt",prompt,"agentIds","agentSnapshots" from "Tasks"'
        agent_df = self.pg.read_sql_to_df(sql)
        # models
        sql = 'select id as "llmId",model,type,"maxTokens",temperature from "LlmConfigs";'
        model_df = self.pg.read_sql_to_df(sql)
        agent_df = agent_df.merge(model_df, left_on="llmId",right_on="llmId", how="left")
        agent_df = agent_df.drop(columns=["llmId"])
        # id,outId,updatedAt,prompt,agentIds,model
        return agent_df

    def load_v2_session(self):
        # select * from "Questions" where "taskId" = 'cmfbeh84500yv18087akmu1z6';
        # select * from "TaskProgresses" where "taskId" = 'cmfbeh84500yv18087akmu1z6';
        sql = '''select "taskId",index,round,step,content,"updatedAt" from "TaskProgresses" where type='agent:lifecycle:memory:added' and "updatedAt" > '2025-09-01';'''
        chat_df = self.pg.read_sql_to_df(sql)
        chat_df = chat_df.sort_values(["taskId","round","index"])
        return chat_df


    def eval_v2(self):
        chat_df = self.load_v2_session()
        group_index = ["taskId","round"]
        records = []
        for (task_id,round),group_df in chat_df.groupby(group_index):
            round = int(round)
            messages = group_df['content'].to_list()
            msg_eval = MessageEval(messages)
            score = msg_eval.compute()
            print(f"task_id: {task_id}/{round} => {score}")
            records.append({"taskId":task_id,"round":round,"score":score,"messages":messages})
        score_df = pd.DataFrame(records)
        # get agent id
        sql = '''select "taskId",round,"primaryAgentId" as "agentId" from "Questions" where "isAgentTrigger" is true and "updatedAt" > '2025-09-01';'''
        question_df = self.pg.read_sql_to_df(sql)
        score_df = score_df.merge(question_df, on=["taskId","round"], how="left")
        agent_df = self.load_v2_agents()
        score_df = score_df.merge(agent_df, on="agentId", how="left")
        return score_df
    
    def save_result(self,result_df:pd.DataFrame):
        self.pg.write_df(result_df,"TaskEval",write_mode="overwrite")

    def run(self):
        result_df = self.eval_v2()
        self.save_result(result_df)

class MessageEval:
    "session内评分,支持动态的多rules价值函数"
    def __init__(self,messages: list):
        self.messages = messages    
        self.score = 0
        self.llm_messages = self.get_messages(["assistant","tool"])

    def __len__(self):
        return len(self.messages)

    def get_messages(self,roles:List[str]) -> list:
        return [message for message in self.messages if message["role"] in roles]
    
    def compute(self):
        self.has_success_rag()
        self.has_success_tools()
        self.has_success_plan()
        self.has_success_sql()
        self.has_success_terminal()
        self.has_success_html()
        return self.score

    def has_success_rag(self) -> bool:
        "具有 rag 调用, 监测通用 rag 调用(function name: rag_query)"
        has_rag = any([msg['message']['name'] == "rag_query" for msg in self.messages if msg['role'] == "tool"])
        self.score += 1 if has_rag else 0
        return has_rag

    def has_success_tools(self) -> bool:
        "具有工具调用, 监测通用工具调用(包含rag的工具调用)"
        has_tool = any([msg['message']['name'] != "rag_query" for msg in self.messages if msg['role'] == "tool"])
        self.score += 1 if has_tool else 0
        return has_tool
    
    def has_success_plan(self):
        """具有计划调用, 监测通用计划调用(function name: plan_query)"""
        has_plan = any([msg['message']['name'] == "plan_query" for msg in self.messages if msg['role'] == "tool"])
        self.score += 1 if has_plan else 0
        return has_plan

    def has_success_sql(self):
        """具有 sql 调用, 监测通用 sql 调用(function name: sql_query)"""
        has_sql = any([msg['message']['name'] == "sql_query" for msg in self.messages if msg['role'] == "tool"])
        self.score += 1 if has_sql else 0
        return has_sql

    def has_success_terminal(self):
        """具有 终端 调用, 监测通用终端调用(function name: terminal_query)"""
        has_terminal = any([msg['message']['name'] == "terminal_query" for msg in self.messages if msg['role'] == "tool"])
        self.score += 1 if has_terminal else 0
        return has_terminal

    def has_success_html(self):
        """具有 html 调用, 监测通用 html 调用(function name: html_query)"""
        has_html = any([msg['message']['name'] == "html_query" for msg in self.messages if msg['role'] == "tool"])
        self.score += 1 if has_html else 0
        return has_html
    
if __name__ == "__main__":
    pg = PostgreSQLData.load_from_env()
    eval_or_rl = EvalOrRL(pg)
    eval_or_rl.run()
        
