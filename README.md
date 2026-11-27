# MD_TreeFree - QGIS plugin

Click a tree in aerial imagery and get an outline that follows the real crown edge.
Left click = capture, click-drag = circle, right click = delete crown under cursor.

## Install in QGIS from this repository (no zip upload needed)

1. QGIS > **Plugins > Manage and Install Plugins > Settings**
2. Click **Add...** and enter
   - Name: `MD_TreeFree`
   - URL: `https://raw.githubusercontent.com/modidev-varkala/md_tree_free/main/plugins.xml`
3. Open the **All** tab, search `MD_TreeFree`, click **Install Plugin**.
4. Later updates appear under the **Upgradeable** tab.

## Updating the plugin (maintainers)

Edit files in `md_tree_free/` (raise `version=` in `md_tree_free/metadata.txt`) and commit.
The GitHub Action rebuilds `md_tree_free.zip` and `plugins.xml` automatically.
Manual rebuild: `python tools/build_repo.py <github-user> <repo> main`
