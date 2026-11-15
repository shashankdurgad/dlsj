# LDraw parts used for the twin's Duplo bricks

Source: the official LDraw parts library, <https://library.ldraw.org/library/official/> (fetched 2026-11-14 by
`python sim/twin/scene/ldraw.py fetch`), files kept unmodified under `parts/` and `p/`.

| Twin kind | LDraw file | Title |
|---|---|---|
| 2x2 | parts/3437.dat | Duplo Brick 2 x 2 |
| 2x4 | parts/3011.dat | Duplo Brick 2 x 4 |
| curved_2x3 | parts/2302b.dat | Duplo Brick 2 x 3 with Curved Top |

plus the sub-parts and primitives they reference (`parts/s/`, `p/`, `p/48/`).

Licence: every file carries `0 !LICENSE Licensed under CC BY 4.0 : see CAreadme.txt`. The LDraw parts library is
licensed under the Creative Commons Attribution 4.0 International licence
(<https://creativecommons.org/licenses/by/4.0/>; contributor agreement: <https://www.ldraw.org/article/349.html>).
Attribution: LDraw.org Parts Library, authors Alex Taylor [anathema], Chris Dee [cwdee], Donald Sutter [technog], James Jessiman, Marc Klein [marckl], Mark Kennedy [mkennedy], Max Martin Richter [MMR1988], Michael Heidemann [mikeheide], Niels Karsdorp [nielsk], Paul Easter [pneaster], Philippe Hurbain [Philo], Steffen [Steffen], Takeshi Takahashi [RainbowDolphin], Tony Hafner [hafhead], Tore Eriksson [Tore_Eriksson], Willy Tschager [Holly-Wood].

Derived files: `../duplo_<kind>.npz` (triangle meshes, Z-up metres, 0.15 mm play per side, 0.4 mm rounded outer edges
on the box bricks) are adaptations made by `sim/twin/scene/ldraw.py` and are under the same licence.
LEGO and DUPLO are trademarks of the LEGO Group, which does not sponsor or endorse this project or LDraw.
