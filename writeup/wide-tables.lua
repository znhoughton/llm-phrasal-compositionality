-- wide-tables.lua
-- Replaces \begin{table} / \end{table} with \begin{table*} / \end{table*}
-- in LaTeX raw blocks, so that tables span both columns in a two-column layout.

function RawBlock(el)
  if el.format == "latex" then
    el.text = el.text:gsub("\\begin{table}", "\\begin{table*}")
    el.text = el.text:gsub("\\end{table}", "\\end{table*}")
    return el
  end
end
