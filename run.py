import pandas as pd
from api.pg_data import PostgreSQLData
from api.rules import LoadRules
import re  # noqa
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser


class EvalOrRL:
    def __init__(self, pg: PostgreSQLData, start_date: str = "2025-09-01"):
        self.pg = pg
        self.start_date = start_date
        self.eval_rules = LoadRules("conf")

    def this_day(self):
        """设置当前日期"""
        self.start_date = datetime.now().strftime("%Y-%m-%d")
        return self

    def load_v2_agents(self):
        sql = 'select id as "agentId","llmId",name as "agentName","updatedAt",responsibilities from "Agents";'
        agent_df = self.pg.read_sql_to_df(sql)
        # models
        sql = (
            'select id as "llmId",model,type,"maxTokens",temperature from "LlmConfigs";'
        )
        model_df = self.pg.read_sql_to_df(sql)
        agent_df = agent_df.merge(
            model_df, left_on="llmId", right_on="llmId", how="left"
        )
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
        sql = (
            'select id as "llmId",model,type,"maxTokens",temperature from "LlmConfigs";'
        )
        model_df = self.pg.read_sql_to_df(sql)
        agent_df = agent_df.merge(
            model_df, left_on="llmId", right_on="llmId", how="left"
        )
        agent_df = agent_df.drop(columns=["llmId"])
        # id,outId,updatedAt,prompt,agentIds,model
        return agent_df

    def load_v2_session(self):
        # select * from "Questions" where "taskId" = 'cmfbeh84500yv18087akmu1z6';
        # select * from "TaskProgresses" where "taskId" = 'cmfbeh84500yv18087akmu1z6';
        sql = f"""select "taskId",index,round,step,content,"updatedAt" from "TaskProgresses" where type='agent:lifecycle:memory:added' and "updatedAt" > '{self.start_date}';"""
        chat_df = self.pg.read_sql_to_df(sql)
        chat_df = chat_df.sort_values(["taskId", "round", "index"])
        return chat_df

    def eval_v2(self):
        chat_df = self.load_v2_session()
        group_index = ["taskId", "round"]
        bar = tqdm(chat_df.groupby(group_index))
        records = []
        for (task_id, round), group_df in bar:
            round = int(round)
            messages = group_df["content"].to_list()
            score = self.eval_rules.score(messages)
            info = f"{task_id}/{round} => {score}"
            bar.set_description(info)
            # records.append({"taskId":task_id,"round":round,"score":score,"messages":messages})
            records.append({"taskId": task_id, "round": round, "score": score})
        score_df = pd.DataFrame(records)
        # get agent id
        sql = f"""select "taskId",round,"primaryAgentId" as "agentId" from "Questions" where "isAgentTrigger" is true and "updatedAt" > '{self.start_date}';"""
        question_df = self.pg.read_sql_to_df(sql)
        score_df = score_df.merge(question_df, on=["taskId", "round"], how="left")
        agent_df = self.load_v2_agents()
        score_df = score_df.merge(agent_df, on="agentId", how="left")
        return score_df

    def save_result(self, result_df: pd.DataFrame):
        self.pg.write_df(result_df, "TaskEval", write_mode="overwrite")

    def run(self):
        print("start eval v2")
        result_df = self.eval_v2()
        print("end eval v2")
        print("start save result. shape: ", result_df.shape)
        self.save_result(result_df)
        print("end save result")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--from_date",
        type=str,
        required=False,
        default="2025-09-01",
        help="从指定的日期开始计算,默认: 2025-09-01",
    )
    parser.add_argument("--today", action="store_true", help="只计算今天的数据")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    s_args = parse_args()
    from_date = datetime.now().strftime("%Y-%m-%d") if s_args.today else s_args.from_date
    pg = PostgreSQLData.load_from_env()
    eval_or_rl = EvalOrRL(pg,from_date)
    eval_or_rl.run()
