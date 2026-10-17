from collections import defaultdict

class S:
    OPEN = "01 专业开场及身份核实"
    FACT = "02 欠款事实确认及情况交互"
    PRESS = "03 充分施压"
    NEGOTIATE = "04 方案协商与签署"
    TRACE = "05 失联联系及线索管理"
    PTP = "06 承诺还款跟进"
    AUTH = "07 外呼授权设计"

SOP_STAGES = [S.OPEN, S.FACT, S.PRESS, S.NEGOTIATE, S.TRACE, S.PTP, S.AUTH]

class I:
    ID_VERIFY = "客户身份核实"
    DELEGATE = "客户代办处理"
    BILL_CHECK = "账单逾期核实沟通"
    ABILITY_EVAL = "逾期还款能力评估"
    WILL_NEGOTIATE = "协商意愿与分期诉求"
    PRESS_STRATEGY = "案件安排施压策略"
    PRESS_LEGAL = "施压与法务风险说明"
    PRESS_ESCALATE = "预警及法务风险升级"
    THIRD_PARTY = "第三方代办意向"
    QUALIFY = "分类资格额度条件"
    MATERIAL = "申请材料及风险解除"
    PLAN_INSTALL = "分期及减免方案"
    PLAN_SETTLE = "一次性结清方案"
    PLAN_REDUCE = "息费减免适用边界"
    PLAN_AGAIN = "单次调解再次协商"
    PLAN_BENEFIT = "方案权益与还款安排"
    SIGN = "协议签署确认"
    REGISTER = "诉求登记与回访"
    CONTACT_FIX = "联系方式地址修复"
    ADDRESS_UPDATE = "地址失效资料更新"
    CONTACT_CHANGE = "联系人变更确认"
    THIRD_CONTACT = "第三人联系转达"
    CALL_MANAGE = "呼叫诉求管理确认"
    KEEP_CONTACT = "保持联系紧急转达"
    PAY_GUIDE = "还款操作代还指引"
    AMOUNT_CONFIRM = "方案选择金额确认"
    VOICE_CONFIRM = "语音确认方案完成"
    SPECIAL_REDUCE = "特殊减免指引"
    AUTH_DESIGN = "外呼授权设计"
    OTHER = "其他"

UNIFIED_INTENTS = [v for k, v in vars(I).items() if not k.startswith('_')]

INTENT_KEYWORDS = {
    I.ID_VERIFY: ["是我", "我就是", "对", "是的", "本人"],
    I.DELEGATE: ["帮他处理", "代替他", "我是他老婆", "受托", "他不在"],
    I.BILL_CHECK: ["知道", "确实", "清楚", "有这笔", "账单不对"],
    I.ABILITY_EVAL: ["没钱", "失业", "还不起", "不想还", "工资没发"],
    I.WILL_NEGOTIATE: ["怎么还", "想还", "商量一下", "可以谈", "分期"],
    I.PRESS_STRATEGY: ["别打了", "不聊", "告你们", "投诉", "报警"],
    I.PRESS_LEGAL: ["起诉", "律师", "上门", "冻结"],
    I.PRESS_ESCALATE: ["我不信", "随便你", "爱咋咋地"],
    I.THIRD_PARTY: ["家人帮还", "朋友代还", "帮我代办", "委托他人"],
    I.QUALIFY: ["符合条件", "可以办", "同意申请", "有资格"],
    I.MATERIAL: ["准备好了", "发过去了", "没带", "晚点提供"],
    I.PLAN_INSTALL: ["分期", "分几期", "减免"],
    I.PLAN_SETTLE: ["一次还清", "结清", "不拖了"],
    I.PLAN_REDUCE: ["减点利息", "免违约金"],
    I.PLAN_AGAIN: ["再商量", "还不了那么多"],
    I.PLAN_BENEFIT: ["怎么还", "去哪还"],
    I.SIGN: ["签了", "同意协议", "按手印"],
    I.REGISTER: ["后续跟进", "会有回复吗"],
    I.CONTACT_FIX: ["搬家了", "换电话了", "不在那了"],
    I.ADDRESS_UPDATE: ["联系不上", "查一下", "没这个人"],
    I.CONTACT_CHANGE: ["不是本人", "打错了", "他不用这号了"],
    I.THIRD_CONTACT: ["他在忙", "我转告他", "别打了"],
    I.KEEP_CONTACT: ["发短信吧", "加微信", "紧急联系人"],
    I.PAY_GUIDE: ["怎么还款", "打钱到哪", "去哪操作"],
    I.AMOUNT_CONFIRM: ["还多少", "金额对", "确认金额"],
    I.VOICE_CONFIRM: ["录音确认", "确认完毕", "搞定了"],
    I.SPECIAL_REDUCE: ["能减免吗", "特殊申请"],
    I.AUTH_DESIGN: ["系统授权", "登记完成", "审批通过"]
}

SOP_FLOW = {
    S.OPEN: { I.ID_VERIFY: {"next": [S.FACT], "seed": ["身份核实", "本人确认"]}, I.DELEGATE: {"next": [S.TRACE], "seed": ["他人代接", "代办"]} },
    S.FACT: { I.BILL_CHECK: {"next": [S.PRESS], "seed": ["账单核实", "逾期金额"]}, I.ABILITY_EVAL: {"next": [S.PRESS], "seed": ["还款能力", "经济困难"]}, I.WILL_NEGOTIATE: {"next": [S.NEGOTIATE], "seed": ["协商意愿", "分期诉求"]} },
    S.PRESS: { I.PRESS_STRATEGY: {"next": [S.PRESS], "seed": ["施压", "后果告知"]}, I.PRESS_LEGAL: {"next": [S.NEGOTIATE], "seed": ["法务风险", "律师函"]}, I.PRESS_ESCALATE: {"next": [S.TRACE], "seed": ["升级预警", "拒绝沟通"]} },
    S.NEGOTIATE: { I.THIRD_PARTY: {"next": [S.NEGOTIATE], "seed": ["第三方代办", "代还"]}, I.QUALIFY: {"next": [S.NEGOTIATE], "seed": ["资格审核", "条件"]}, I.MATERIAL: {"next": [S.NEGOTIATE], "seed": ["申请材料", "风险解除"]}, I.PLAN_INSTALL: {"next": [S.NEGOTIATE], "seed": ["分期方案", "减免"]}, I.PLAN_SETTLE: {"next": [S.NEGOTIATE], "seed": ["一次性结清", "全额"]}, I.PLAN_REDUCE: {"next": [S.NEGOTIATE], "seed": ["息费减免", "违约金"]}, I.PLAN_AGAIN: {"next": [S.PRESS], "seed": ["再次协商", "方案异议"]}, I.SIGN: {"next": [S.PTP], "seed": ["协议签署", "确认"]}, I.REGISTER: {"next": [S.PTP], "seed": ["诉求登记", "回访"]} },
    S.TRACE: { I.CONTACT_FIX: {"next": [S.TRACE], "seed": ["联系方式修复", "新号码"]}, I.ADDRESS_UPDATE: {"next": [S.TRACE], "seed": ["地址更新", "资料维护"]}, I.CONTACT_CHANGE: {"next": [S.TRACE], "seed": ["联系人变更", "核实"]}, I.THIRD_CONTACT: {"next": [S.TRACE], "seed": ["第三人转达", "家属"]}, I.KEEP_CONTACT: {"next": [S.TRACE], "seed": ["保持联系", "紧急转达"]}, I.ID_VERIFY: {"next": [S.FACT], "seed": ["重新联系上本人", "核实"]} },
    S.PTP: { I.PAY_GUIDE: {"next": [S.PTP], "seed": ["还款指引", "操作流程"]}, I.AMOUNT_CONFIRM: {"next": [S.PTP], "seed": ["金额确认", "账单"]}, I.SPECIAL_REDUCE: {"next": [S.PTP], "seed": ["特殊减免", "申请"]}, I.VOICE_CONFIRM: {"next": [S.AUTH], "seed": ["语音确认", "承诺完成"]} },
    S.AUTH: { I.AUTH_DESIGN: {"next": [S.AUTH], "seed": ["外呼授权", "系统登记"]} }
}

STAGE_GATED_INTENTS = {stage: list(flow.keys()) for stage, flow in SOP_FLOW.items()}
TRANSITION_ADJ = defaultdict(list)
for stage, flow in SOP_FLOW.items():
    for intent, data in flow.items():
        for nxt in data["next"]:
            TRANSITION_ADJ[(stage, intent)].append(nxt)

SEED_TOPIC_LIST = []
for stage in SOP_STAGES:
    stage_seeds = []
    for intent, data in SOP_FLOW.get(stage, {}).items():
        stage_seeds.extend(data["seed"])
    SEED_TOPIC_LIST.append(stage_seeds)

# --- Prompts 常量 ---
SESSION_REASONS = ["失业", "收入减少", "医疗费用高", "生意失败", "家庭变故", "多头借贷", "生活开支高", "发薪日延迟", "账单争议", "疑似欺诈", "遗忘漏缴"]
SESSION_TACTICS = ["核对身份", "隐私披露", "同理心安抚", "确认困难", "欠款告知", "原因探查", "还款能力评估", "法务告知", "引导分期", "引导减免", "抛出替代资金方案", "锁定PTP"]
SESSION_OBJECTIONS = ["否认欠款", "非本人", "质疑金额", "质疑利息", "失业", "收入减少", "暂时无力偿还", "要求延期", "要求分期", "要求减免", "拒绝还款", "情绪激动", "威胁投诉"]
SESSION_COMPLIANCE = ["本人身份确认完成", "非本人终止", "隐私合规披露", "威胁话术禁止", "虚假陈述禁止", "PTP要素完整"]
SESSION_OUTCOMES = ["全额承诺还款(PTP)", "部分承诺还款(PTP)", "达成减免方案", "达成分期方案", "协商未成(约定再拨)", "协商未成(拒绝还款)", "转交法务", "投诉升级", "失联"]
SESSION_EMOTIONS = ["平静", "焦虑", "抵触", "愤怒", "合作", "无奈"]
PAIR_ACTIONS = ["身份核实", "欠款告知", "原因探查", "财务探询", "同理心安抚", "施压警告", "法务告知", "引导分期", "引导减免", "抛出替代资金方案", "锁定PTP", "确认还款日期", "信息更新", "约定再拨"]
PAIR_KEYWORDS = ["欠款金额", "还款金额", "分期期数", "短期延期", "医疗费用", "失业状态", "家庭变故", "还款渠道", "联系方式"]
PAIR_TAGS = ["情绪激动", "情绪平稳", "积极配合", "消极应对", "抗拒沟通", "有还款意愿", "无还款意愿", "有投诉风险", "话术打断", "信息模糊", "承诺明确", "方案清晰", "合规正常", "敏感信息", "法律风险", "高投诉风险", "失联风险"]
PAIR_RISK_LEVELS = ["High", "Low", "None"]
PAIR_RISK_TYPES = ["违规施压", "辱骂威胁", "虚假陈述", "第三方披露", "承诺未兑现", "减免权限不清", "法务威胁不当"]
PAIR_EMOTIONS = ["平静", "焦虑", "抵触", "愤怒", "合作", "无奈"]