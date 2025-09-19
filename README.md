
# issues

1. 多个rules, 让llm选择应该使用哪些
2. python 的 rules 独立出来, 方便配置
3. 大模型(使用比较快的)的打分

```sql
-- select * from "V2AgentMessages" order by "created_at" limit 100;
-- select id,name,"llmId","updatedAt",responsibilities from "Agents"
-- select * from "Agents"
-- select * from "LlmConfigs";
-- select id,"outId","llmId","updatedAt",prompt,"agentIds","agentSnapshots" from "Tasks" where "id"='cmfbeh84500yv18087akmu1z6'
-- select * from "TaskProgresses" where "taskId" = 'cmfbeh84500yv18087akmu1z6' and type='agent:lifecycle:memory:messagesSnapshot';
select * from "TaskProgresses" where "taskId" = 'cmfbeh84500yv18087akmu1z6' and type='agent:lifecycle:memory:added';
-- select * from "Questions" where "taskId" = 'cmfbeh84500yv18087akmu1z6';
-- select "taskId",index,round,step,content,"updatedAt" from "TaskProgresses" where "taskId" = 'cmfbeh84500yv18087akmu1z6' and type='agent:lifecycle:memory:added';
```
