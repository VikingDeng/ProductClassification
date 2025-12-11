## 简单的指南
1. scripts中有后台运行和可视化脚本，服务器上跑要做正向代理，命令已经写在vis.sh中
2. configs中的配置可供参考，添加新模型的时候往src/models中添加,一般来说分为freeze的backbone和trainable的head
3. 注册modol、data-trainsform的时候，除了实现对应的代码，还需要在模块的__init__.py中注册
4. 目前的默认runner策略是kfold交叉验证，然后做投票集成