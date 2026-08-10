# Design System & UI Guidelines

## Design Tokens
- **Theme**: Dark Mode only.
- **Background**: `Zinc-950` (`#09090b`)
- **Card Background/Surface**: `Zinc-900` (`#18181b`)
- **Borders/Dividers**: `Zinc-800` (`#27272a`)
- **Foreground Text**: `Zinc-100` (`#f4f4f5`) / Muted Text: `Zinc-400` (`#a1a1aa`)
- **Semantic Highlights (Retained Tokens)**: `Emerald-500` (`#10b981`) text highlight or subtle green background with green glow.
- **Pruned Tokens (Filler)**: `Rose-500` (`#f43f5e`) color with a clean strikethrough animation/style.

## Typography
- **Primary Sans-Serif**: `Inter` or `Outfit` (for layout, headers, and dashboard metrics).
- **Monospace Code/Token Font**: `JetBrains Mono` (for text compression comparison panes).

## Components & Layouts
- **Shadcn UI Dashboard Layout**: Sleek sidebar navigation, clean metrics cards.
- **Split-Pane Viewer**: Left pane displays original retrieved text; Right pane displays the compressed text stream. Or, a unified view highlighting retained tokens in `emerald` and pruned tokens in `rose` with strikethrough.
- **Metrics Dashboard**: Display cards for:
  - Context Compression Ratio (%)
  - Latency Saved (ms)
  - Estimated Cost Drop ($)
