# 还没提上去的补丁

`patches/<repo>/` 下的补丁按 P9 都必须带 `Upstream-Issue:` / `Upstream-PR:` 头，
`tests/unit/test_patches_and_status.py` 会强制这一点。

这个目录放**已经写好、还没对外提交**的补丁：正文写完了，等一个人点头再发。
提交之后把补丁移回上一级并补上两个 header，测试会重新覆盖它。

放在这里的补丁，`docs/issues/STATUS.md` 里那一行必须写"未提交"。
