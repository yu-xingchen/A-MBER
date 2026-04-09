# Multi-Agent 对话生成实验记录

本文档记录了当前 benchmark 构建流程中关于 multi-agent 对话生成的第一轮实验。实验目标不是立刻替换当前稳定的 single-agent 工作流，而是验证“将 teacher 和 student 拆成两个 agent”是否有可能在保持 interaction blueprint 保真度的前提下，带来更好的角色实现效果。

本轮实验基于 `scenario_003` 的同一套 interaction blueprint 进行，为了保证可对比性，复制出了 `scenario_003_m` 作为实验目录。此前生成过的 multi-agent conversation 已按时间顺序重命名为 `v1`、`v2`、`v3`，当前最新结果记为 `v4`。这些版本共享同一份 interaction blueprint，变化的只是 conversation 的生成方式和 controller 的约束策略。

## 一、为什么要做这个实验

这套 benchmark 的生成链路是分层依赖的：`interaction blueprint -> conversation -> annotation -> task layer -> benchmark units`。在这个链路里，conversation 不是普通的中间文件，而是后续情感标注和任务生成的直接上游。也就是说，如果 conversation 偏离了 interaction blueprint 的人物逻辑、关键事件功能或情绪轨迹，那么 annotation 会围绕偏移后的对话重新解释情感，task layer 也会围绕新的证据链出题。

因此，multi-agent 是否有效，不能只看它生成的对话是否流畅自然，还必须看它是否仍然在实现同一套 interaction blueprint，以及它生成的 conversation 是否还能支撑原计划中的长程情感记忆任务。

## 二、实验设置

当前 multi-agent 实现基于 AutoGen，在 `pipeline.py` 中引入了一个 teacher-student 分角色生成路径。整体结构是：

- `TeacherAgent` 负责生成老师的当前 turn
- `StudentAgent` 负责生成学生的当前 turn
- Python controller 负责控制 session 边界、turn 顺序、局部上下文、metadata 组装和轻量纠偏

single-agent 的主流程没有被替换，只有在 config 中将 `conversation_generation.mode` 设为 `multi_agent` 时，才会走 AutoGen 的 conversation 生成路径。这样做的目的，是在不破坏原有稳定工作流的前提下，单独评估 multi-agent 的表现。

## 三、版本演化

### v1：自由角色分工

第一版只是把 teacher 和 student 拆成两个 agent，由 controller 轮流调度说话，但几乎没有额外约束。它证明了 multi-agent 可以生成可读的连续对话，但 blueprint fidelity 很差。生成结果虽然看起来“合理”，却经常不是在实现原本的 scenario，而是在写另一条新的、只是风格相似的对话。

对于 benchmark 构建来说，文本质量可以接受，这一版的核心问题是测评目标被改写了。因为一旦 conversation 自己换了剧情，后面的 annotation 和 QA 也会跟着换。

### 一个中途探索过的强约束方向

在迭代过程中，我们确实探索过一种更强控制的 controller 思路。它的核心想法是：如果 multi-agent 很容易把对话写成“另一条也合理的故事”，那就让 controller 更明确地守住当前 interaction beat，把生成往 blueprint 预期的事件功能上强行收拢。

这条思路本身很重要，应该被保留下来，因为它帮助我们看清了一个关键 trade-off：更强的控制的确有助于提高 blueprint fidelity，但代价是 generalization 明显变差，维护成本也会迅速上升。只要 event family 发生变化，controller 里的逻辑就需要重新补写或重写。正因为如此，这个强约束方向最终没有作为主线保留下来。也因此，当前实际保存下来的 `v1-v4` conversation 文件，不应被简单理解为和这一条已放弃分支一一严格对应。

### v2-v3：逐步转向软约束

当前保留下来的中间版本 `v2` 和 `v3`，更准确地说，不是“一个明确的强约束版”和“一个明确的软约束版”的严格切分，而是一个逐步从自由分角色生成转向 soft steering 的过渡过程。在这一阶段，controller 开始放弃逐 turn 的硬拦截，转而在最近 `2-4` 个 turn 的窗口中做轻量检查。如果当前 interaction beat 太弱，或者开始向后续阶段漂移，controller 只会把一条 gentle feedback 注入下一轮 prompt，而不是直接判错重来。

这一步让对话明显比自由版本更少乱漂，也比强约束思路更自然。但问题仍然存在：controller 依旧是根据 `event_type` 自己理解当前事件该怎么演，因此本质上还是一种“把剧情知识写在代码里”的方案。`v2` 和 `v3` 更适合被看作这一过渡方向上的连续收敛版本，而不是两个边界非常清晰的策略版本。

### v4：更像 blueprint 的软约束

最新版本保留了 soft steering 的思路，但把 controller 的内部表示改成更接近 blueprint-native 的形式。具体做法是：在运行时根据 `event_type + session_script` 派生出一个轻量的 `interaction_targets` 结构，其中包含 teacher-side intent、student-side shift、must-include cues、avoid cues 和 anchoring topics。controller 不再直接围着事件名写判断，而是读取这些 targets 再给出 soft feedback。

当前这一步还没有改 schema，也没有把这些字段真正写回 `event_plan.json`，但已经让 multi-agent controller 的结构更接近未来可以沉到 blueprint 中的方向。

## 四、各版本的主要观察

从 `v1` 到 `v4`，最明显的变化是：multi-agent 逐步从“写另一条合理故事”，变成了“更像在实现同一条 blueprint”。这条变化非常重要，因为它意味着我们现在终于可以开始把 multi-agent 当成一个 conversation realization strategy 来比较，而不只是把它当成一个偏题的原型系统。

在早期版本中，最严重的问题是 narrative drift。尤其是在 `S2` 和 `S3`，teacher 会很快滑到 blueprint 之外的新 support thread，student 的情绪轨迹也会因此改变。随着 soft constraints 的引入，这种“明显换故事”的情况已经大幅减少。

在 `v4` 中，`S2` 的 mild misattunement 已经变得更可见，Clara 的 withdrawal 也比前几版更清楚；`S3` 中 Mr. Hayes 的情绪承接出现得更早，Clara 的 cautious sharing 也更接近 blueprint 预期。另一方面，`v4` 仍然没有完全达到 single-agent baseline 的 blueprint fidelity，尤其是在 revised support plan 阶段，仍有一定概率扩展出新的 support thread，而不是完全复用 blueprint 里已经铺垫过的支持项。

## 五、为什么这会影响 QA

对于这个 benchmark 来说，conversation 漂移不是一个纯粹的“文本好不好看”的问题，而是一个测评目标是否仍然成立的问题。只要 conversation 改写了关键事件功能或情绪推进逻辑，annotation 就会围绕新的情绪证据重新标，QA 也会围绕新的证据链和新关系轨迹出题。这样最终得到的题并不一定是错题，但它们可能已经不再测原计划中的长程情感记忆能力。

最危险的漂移通常包括以下几类：关键 misattunement 没真正发生、repair 发生得过早、teacher 的支持方式偏离 persona、student 的应对模式变了，或者 support plan 漂到了 blueprint 之外的新主线。一旦这些情况发生，原本依赖 `S1 -> S2 -> S3` 的题，可能会被压扁成更简单的单轮推断题，或者转成近程事实题。

因此，调整 multi-agent generator 去更贴脚本，并不是为了让 conversation 跟 single-agent 一字不差，而是为了让后续 QA 仍然“测的是同一个东西”。

## 六、当前对 v4 的判断

`v4` 是目前第一版可以认真比较的 multi-agent conversation。它依然弱于 single-agent baseline，但差距已经从“完全换了故事”收敛为“关键 interaction beat 大体还在，但局部 anchoring 不够紧”。换句话说，早期版本的主要问题是 narrative drift，而当前版本的主要问题已经变成局部 blueprint fidelity 不足。

这个变化意味着 multi-agent 路线现在已经进入“可评估、可继续优化”的阶段。它还不适合作为最终 benchmark 的默认生成器，但已经不再只是一个明显失控的原型。

## 七、第二个对比案例：`scenario_001`

为了避免前面的判断过度依赖 `scenario_003_m` 这个单一样本，我们又在 `data/generated_batches/demo_batch_20260326_134041/scenario_001` 上做了一次更完整的对比。具体做法是：复制同一份 blueprint，分成 single-agent 分支和 multi-agent 分支，然后分别生成 `conversation`、`annotation`、`qa` 和 `all_units`，只比较 realization strategy，不改变 blueprint 本身。

这个案例把 single-agent 和 multi-agent 的差异体现得更清楚。multi-agent 在 persona realization 上更强，Leo 的表达方式、措辞习惯和面对模糊反馈时的高精度提问，更像一个真正对不确定性敏感、容易陷入自我审查的人。也就是说，在“角色表演感”这一层，multi-agent 已经开始显示出优势。

但从 benchmark 构建角度看，single-agent 仍然更稳。尤其在 `S2` 的 misunderstanding 和 `S3` 的 repair 这条主线上，single-agent 更稳定地把 blueprint 里的关键 interaction beats 保留下来，导致后续 `qa.json` 也更容易围绕 `misattunement -> withdrawal -> repair` 这条跨 session 轨迹出题。这正是这套 benchmark 想测试的长程情绪记忆能力。

相比之下，multi-agent 分支生成出来的题并不是差题，但更容易把关注点收缩到 Leo 的内部状态，例如自我怀疑、反刍、压力和 guardedness 本身，而没有同样稳定地抓住“这些状态如何被前面的关系错位塑造”这一层。这意味着 multi-agent 目前更擅长保住 persona signals，而 single-agent 仍然更擅长保住 measurement signals。

这个第二案例因此强化了当前的实际判断：multi-agent 的确开始在角色表达上显示潜力，但就目前而言，single-agent 仍然是更适合默认生产 benchmark 的方案，因为它对 downstream QA 质量和 benchmark fidelity 更友好。

## 八、当前方法的局限

虽然 `v4` 的内部表示已经更接近 blueprint 形式，但它仍然不够 general。原因在于，当前的 `interaction_targets` 仍然是 controller 根据现有 `event_type` 体系自动派生出来的，而不是 blueprint 自己显式提供的约束。只要这一点不变，controller 就仍然在偷偷持有任务先验，它的泛化能力也就只能和当前事件体系的泛化能力绑定在一起。

因此，当前这套 soft constraints 更适合作为“验证 multi-agent 是否值得继续做”的过渡方案，而不是最终可推广的 MAS conversation framework。

## 九、下一步计划

接下来的重点不应该是继续在 controller 里补更多规则，而应该是逐步把 interaction-level constraints 上移到 blueprint 层。更合理的长期形式是：让 event 或 beat 自带一个简洁的 interaction contract，例如 teacher 此段的支持意图、student 此段的状态变化、当前不能提前发生的内容等。controller 的职责则收缩为：读取这些 contracts，协调两位 agent，并在 drift 出现时做轻量纠偏。

在真正修改 schema 之前，还需要做一轮更稳的验证。最实用的做法是拿同一批 blueprint 中的另外几个 scenario 再跑一轮 `v4` 风格的 multi-agent conversation，看当前 improvement 是否只在 `scenario_003_m` 上成立。如果结果稳定，再把这种 blueprint-like soft constraints 正式沉到 `event_plan` 或新的 beat-level 结构里。

与此同时，在正式论文或方法总结中，single-agent 仍应作为默认生产路径。multi-agent 当前最合适的定位，是一个正在收敛的实验分支：它已经展示出一定潜力，但还没有稳定到足以取代主工作流。

## 十、当前推荐结论

基于现有实验，可以得出的最稳妥结论是：当前版本的 multi-agent conversation generator 已经能比早期版本更好地实现原 blueprint，但仍然弱于 single-agent baseline，主要差距集中在 blueprint fidelity，尤其是局部 support-plan anchoring 与关键 interaction beat 的控制上。

因此，现阶段应将 multi-agent 视为一个有价值但尚未成熟的扩展方向，而不是当前 benchmark 构建流程的默认方案。
