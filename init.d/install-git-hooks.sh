#!/usr/bin/env bash
# Injected at every session start to link .githooks/pre-commit

shopt -s nullglob
for root in "${RENKU_MOUNT_DIR:-}" "${RENKU_WORKING_DIR:-}" /home/renku/work; do
	[[ -n $root && -d $root ]] || continue
	# in every repo
	for git_dir in "$root"/.git "$root"/*/.git; do
		[[ -d $git_dir ]] || continue
		repo=${git_dir%/.git}
		hook=$git_dir/hooks/pre-commit
		[[ -f $repo/.githooks/pre-commit ]] || continue
		[[ -e $hook || -L $hook ]] && continue
		git -C "$repo" config core.hooksPath >/dev/null 2>&1 && continue
		mkdir -p "$git_dir/hooks" 2>/dev/null &&
			ln -s ../../.githooks/pre-commit "$hook" 2>/dev/null &&
			echo "install-git-hooks: enabled in $repo" >&2
	done
done
exit 0
