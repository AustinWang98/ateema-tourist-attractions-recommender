# Design System

## Overview

ChicagoDoes Recommender uses a cinematic dark product shell after onboarding, matching the first-view hero while preserving product UI clarity. The visual system is dark glass panels over city media, compact controls, media-forward recommendation cards, and a restrained semantic palette.

## Palette

- Background: deep blue-black OKLCH neutrals around hue 232.
- Elevated surfaces: translucent dark navy glass with subtle light borders.
- Primary action: Ateema amber.
- Catalog metadata: cyan/blue tints.
- Hot/trending states: coral and amber.
- Text: high-contrast cool white for primary content, blue-tinted muted text for secondary content.

## Typography

Use Inter as the product UI family. Keep headings compact and confident, avoid oversized display type after the landing hero, and keep labels dense but readable.

## Components

- Panels: dark translucent glass, 12px radius, light hairline border, no wide decorative shadow.
- Cards: media-first dark tiles with light text and compact tags.
- Controls: dark translucent inputs/selects with consistent borders and visible focus rings.
- Chips: pill controls using dark inactive states and semantic active states.
- Pager buttons: dark circular controls with amber hover/active affordances.

## Layout

The workspace is a two-column product tool: optional trip profile sidebar plus results canvas. Results cards maintain stable maximum widths and a consistent carousel grid. Spacing uses a tight product rhythm, with stronger separation between panel header, pager metadata, and content grid.

## Motion

Use short 150-250ms transitions for hover, focus, and state changes only. Avoid decorative page-load choreography.
