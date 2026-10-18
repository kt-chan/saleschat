class S:
    OPEN = "01 专业开场及身份核实"; FACT = "02 欠款事实确认及情况交互"; PRESS = "03 充分施压"
    NEGOTIATE = "04 方案协商与签署"; TRACE = "05 失联联系及线索管理"; PTP = "06 承诺还款跟进"; AUTH = "07 外呼授权设计"

SOP_STAGES = [S.OPEN, S.FACT, S.PRESS, S.NEGOTIATE, S.TRACE, S.PTP, S.AUTH]
class I:
    ID_VERIFY = "客户身份核实"; DELEGATE = "客户代办处理"; BILL_CHECK = "账单逾期核实沟通"
    ABILITY_EVAL = "逾期还款能力评估"; WILL_NEGOTIATE = "协商意愿与分期诉求"; PRESS_STRATEGY = "案件安排施压策略"
    PRESS_LEGAL = "施压与法务风险说明"; PRESS_ESCALATE = "预警及法务风险升级"; THIRD_PARTY = "第三方代办意向"
    QUALIFY = "分类资格额度条件"; MATERIAL = "申请材料及风险解除"; PLAN_INSTALL = "分期及减免方案"
    PLAN_SETTLE = "一次性结清方案"; PLAN_REDUCE = "息费减免适用边界"; PLAN_AGAIN = "单次调解再次协商"
    SIGN = "协议签署确认"; REGISTER = "诉求登记与回访"; CONTACT_FIX = "联系方式地址修复"
    ADDRESS_UPDATE = "地址失效资料更新"; CONTACT_CHANGE = "联系人变更确认"; THIRD_CONTACT = "第三人联系转达"
    KEEP_CONTACT = "保持联系紧急转达"; PAY_GUIDE = "还款操作代还指引"; AMOUNT_CONFIRM = "方案选择金额确认"
    VOICE_CONFIRM = "语音确认方案完成"; SPECIAL_REDUCE = "特殊减免指引"; AUTH_DESIGN = "外呼授权设计"; OTHER = "其他"

# 唯一的意图真理源（同时用于意图、动作、策略、异议）
UNIFIED_INTENTS = [v for k, v in vars(I).items() if not k.startswith('_')]

SOP_FLOW = {
    S.OPEN:      [I.ID_VERIFY, I.DELEGATE],
    S.FACT:      [I.BILL_CHECK, I.ABILITY_EVAL, I.WILL_NEGOTIATE],
    S.PRESS:     [I.PRESS_STRATEGY, I.PRESS_LEGAL, I.PRESS_ESCALATE],
    S.NEGOTIATE: [I.THIRD_PARTY, I.QUALIFY, I.MATERIAL, I.PLAN_INSTALL, I.PLAN_SETTLE, I.PLAN_REDUCE, I.PLAN_AGAIN, I.SIGN, I.REGISTER],
    S.TRACE:     [I.CONTACT_FIX, I.ADDRESS_UPDATE, I.CONTACT_CHANGE, I.THIRD_CONTACT, I.KEEP_CONTACT, I.ID_VERIFY],
    S.PTP:       [I.PAY_GUIDE, I.AMOUNT_CONFIRM, I.SPECIAL_REDUCE, I.VOICE_CONFIRM],
    S.AUTH:      [I.AUTH_DESIGN]
}

SOP_TRANSITIONS = {
    (S.OPEN, I.ID_VERIFY): S.FACT, (S.OPEN, I.DELEGATE): S.TRACE,
    (S.FACT, I.BILL_CHECK): S.PRESS, (S.FACT, I.ABILITY_EVAL): S.PRESS, (S.FACT, I.WILL_NEGOTIATE): S.NEGOTIATE,
    (S.PRESS, I.PRESS_STRATEGY): S.PRESS, (S.PRESS, I.PRESS_LEGAL): S.NEGOTIATE, (S.PRESS, I.PRESS_ESCALATE): S.TRACE,
    (S.NEGOTIATE, I.THIRD_PARTY): S.NEGOTIATE, (S.NEGOTIATE, I.QUALIFY): S.NEGOTIATE, (S.NEGOTIATE, I.MATERIAL): S.NEGOTIATE, (S.NEGOTIATE, I.PLAN_INSTALL): S.NEGOTIATE, (S.NEGOTIATE, I.PLAN_SETTLE): S.NEGOTIATE, (S.NEGOTIATE, I.PLAN_REDUCE): S.NEGOTIATE, (S.NEGOTIATE, I.PLAN_AGAIN): S.PRESS, (S.NEGOTIATE, I.SIGN): S.PTP, (S.NEGOTIATE, I.REGISTER): S.PTP,
    (S.TRACE, I.CONTACT_FIX): S.TRACE, (S.TRACE, I.ADDRESS_UPDATE): S.TRACE, (S.TRACE, I.CONTACT_CHANGE): S.TRACE, (S.TRACE, I.THIRD_CONTACT): S.TRACE, (S.TRACE, I.KEEP_CONTACT): S.TRACE, (S.TRACE, I.ID_VERIFY): S.FACT,
    (S.PTP, I.PAY_GUIDE): S.PTP, (S.PTP, I.AMOUNT_CONFIRM): S.PTP, (S.PTP, I.SPECIAL_REDUCE): S.PTP, (S.PTP, I.VOICE_CONFIRM): S.AUTH,
    (S.AUTH, I.AUTH_DESIGN): S.AUTH
}

# 仅保留真正有区分度的维度
SESSION_REASONS = ["失业", "收入减少", "医疗费用高", "生意失败", "家庭变故", "多头借贷", "生活开支高", "发薪日延迟", "账单争议", "疑似欺诈", "遗忘漏缴"]
SESSION_EMOTIONS = ["平静", "焦虑", "抵触", "愤怒", "合作", "无奈"]
SESSION_COMPLIANCE = ["本人身份确认完成", "非本人终止", "隐私合规披露", "威胁话术禁止", "虚假陈述禁止", "PTP要素完整"]
SESSION_OUTCOMES = ["全额承诺还款(PTP)", "部分承诺还款(PTP)", "达成减免方案", "达成分期方案", "协商未成", "转交法务", "投诉升级", "失联"]

PAIR_KEYWORDS = ["欠款金额", "还款金额", "分期期数", "短期延期", "医疗费用", "失业状态", "家庭变故", "还款渠道"]
PAIR_RISK_LEVELS = ["High", "Low", "None"]
PAIR_RISK_TYPES = ["违规施压", "辱骂威胁", "虚假陈述", "第三方披露", "承诺未兑现", "法务威胁不当"]
PAIR_EMOTIONS = ["平静", "焦虑", "抵触", "愤怒", "合作", "无奈"]