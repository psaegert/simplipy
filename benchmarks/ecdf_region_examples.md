# Region examples — acj-4-3 SOUND, mu ratios (64k nv corpus)

## ratio < 0.9  (2073 rows, 3.16%)

- **row 5995** | mu 129 -> 113 (ratio 0.876) | nodes 27 -> 18
  - orig: `inv sinh * x2 * 5 - + log exp / x2 x2 * x2 x2 / pow / x2 x2 3 + x2 rootn x2 2`
  - simp: `1/sinh(5*x2*(x2^2 - 1/(x2 + x2^(0.5)) + 1))`
- **row 8296** | mu 62 -> 54 (ratio 0.871) | nodes 21 -> 9
  - orig: `* pow * x10 / cos - x10 + asinh x10 x10 x10 5 / * * 5 x10 x10 x10`
  - simp: `5*x10*cos(asinh(x10))^5`
- **row 11875** | mu 114 -> 66 (ratio 0.579) | nodes 28 -> 8
  - orig: `- atanh / x4 x4 + * + + - * 4 x2 x13 x4 tan / x2 x4 + * inv x2 x2 / x15 x15 x13`
  - simp: `-2*tan(x2/x4) + inf`
- **row 16889** | mu 349 -> 213 (ratio 0.610) | nodes 31 -> 14
  - orig: `* * 4 x16 - * x8 / * * * x16 <constant> / x16 / <constant> * / x16 x8 / x8 x8 <constant> neg * x16 x8 cos x8`
  - simp: `-4*x16*(cos(x8) + <constant>*x16^2/x8)`
- **row 31610** | mu 141 -> 125 (ratio 0.887) | nodes 28 -> 18
  - orig: `+ - + - + + * 2 - x1 x1 x8 x14 tanh - x8 x14 * / / + x8 x14 x8 5 x14 x14 x8`
  - simp: `2*x8 + tanh(x14 - x8) + 0.2*x14*(x14 + x8)/x8`
- **row 33128** | mu 700 -> 556 (ratio 0.794) | nodes 31 -> 27
  - orig: `* * - * / + + x7 x7 x2 - * * / <constant> * <constant> <constant> / x15 x6 x15 * x7 x6 asinh x2 <constant> cos <constant> x2`
  - simp: `<constant>*x2*(asinh(x2)*(x2 + 2*x7)/(<constant>*x15^2/x6 - x6*x7) + <constant>)`
- **row 33491** | mu 470 -> 326 (ratio 0.694) | nodes 30 -> 31
  - orig: `cos + - - pow * * + x17 / x10 x4 / x8 5 x17 4 asin / * 5 asinh cosh x17 x10 - <constant> x4 inv <constant>`
  - simp: `cos(x4 + asin(-5*asinh(cosh(x17))/x10) + 0.0016*x17^4*x8^4*(x17 + x10/x4)^4 + <constant>)`
- **row 37611** | mu 469 -> 322 (ratio 0.687) | nodes 23 -> 13
  - orig: `+ - x14 * x14 * * * 2 x12 pow <constant> 5 / * * x12 x14 <constant> x12 / x12 <constant>`
  - simp: `x14 + <constant>*x12 + <constant>*x12*x14^2`
- **row 38039** | mu 219 -> 185 (ratio 0.845) | nodes 34 -> 25
  - orig: `+ + - pow + / - + atanh / x8 x8 x11 x16 x16 x8 4 * 3 x11 - + / x16 x11 x11 x8 - rootn x11 3 * 2 x16`
  - simp: `-2*x11 - 2*x16 - x8 + rootn(x11, 3) + (x8 + inf/x16)^4 + x16/x11`
- **row 41818** | mu 378 -> 242 (ratio 0.640) | nodes 21 -> 15
  - orig: `+ * / + exp x1 x1 sin / <constant> * - x1 + x1 <constant> x7 x1 + x14 x14`
  - simp: `2*x14 + x1*(x1 + exp(x1))/sin(<constant>/x7)`
- **row 41924** | mu 815 -> 669 (ratio 0.821) | nodes 34 -> 28
  - orig: `* x4 * x15 / x8 - x3 * + atanh <constant> + x8 x8 + * + * 5 <constant> rootn <constant> 3 rootn x14 5 + + * x8 <constant> <constant> x3`
  - simp: `x15*x4*x8/(x3 - (x3 + <constant>*x8 + <constant>*rootn(x14, 5) + <constant>)*(2*x8 + atanh(<constant>)))`
- **row 64053** | mu 470 -> 334 (ratio 0.711) | nodes 31 -> 25
  - orig: `+ x6 * - * <constant> / / + sinh x7 x8 x9 <constant> x3 log rootn / - x6 / * 4 / x13 5 2 - x16 x13 2`
  - simp: `x6 - log(((0.4*x13 - x6)/(x13 - x16))^(0.5))*(x3 + <constant>*(x8 + sinh(x7))/x9)`

## 0.9 <= ratio < 1  (11296 rows, 17.24%)

- **row 17110** | mu 101 -> 93 (ratio 0.921) | nodes 14 -> 15
  - orig: `+ x4 - acosh x8 + sinh pow x8 4 sin / x4 2`
  - simp: `x4 + acosh(x8) + sin(-0.5*x4) + sinh(-x8^4)`
- **row 18492** | mu 372 -> 350 (ratio 0.941) | nodes 17 -> 19
  - orig: `- * * + rootn <constant> 5 x5 x6 pow / / <constant> x3 x10 4 x1`
  - simp: `-x1 + <constant>*x6*(x5 + <constant>)/(x10^4*x3^4)`
- **row 19471** | mu 298 -> 282 (ratio 0.946) | nodes 23 -> 24
  - orig: `+ - - x6 / x8 <constant> sin / / x8 + x15 x15 cosh x5 * acosh exp + x5 x15 x10`
  - simp: `x6 + sin(-0.5*x8/(x15*cosh(x5))) + <constant>*x8 + x10*acosh(exp(x15 + x5))`
- **row 21159** | mu 525 -> 514 (ratio 0.979) | nodes 27 -> 22
  - orig: `pow + / pow pow x9 5 2 pow * <constant> log tanh / <constant> * * / x3 x3 + x7 x2 x9 4 <constant> 5`
  - simp: `(<constant>*x9^10/log(tanh(<constant>/(x9*(x2 + x7))))^4 + <constant>)^5`
- **row 27106** | mu 229 -> 221 (ratio 0.965) | nodes 30 -> 30
  - orig: `neg - rootn * x2 / atanh / + pow x8 4 * x13 x5 2 exp / x1 x5 3 - rootn / x2 x15 4 rootn x7 5`
  - simp: `-rootn(x7, 5) - rootn(x2*atanh(0.5*(x8^4 + x13*x5))*exp(-x1/x5), 3) + (x2/x15)^(0.25)`
- **row 29035** | mu 240 -> 232 (ratio 0.967) | nodes 33 -> 33
  - orig: `+ + / - rootn + cos x16 x5 5 x4 x7 - - x12 + x2 x4 x11 / / * x2 / x8 exp * 3 x9 5 + x15 x10`
  - simp: `-x11 + x12 - x2 - x4 + 0.2*x2*x8*exp(-3*x9)/(x10 + x15) - (x4 - rootn(x5 + cos(x16), 5))/x7`
- **row 32703** | mu 241 -> 229 (ratio 0.950) | nodes 23 -> 17
  - orig: `asin / * * / rootn rootn x5 3 5 / <constant> 3 tanh * x14 + x11 x4 * 4 x10 3`
  - simp: `asin(<constant>*x10*rootn(rootn(x5, 3), 5)*tanh(x14*(x11 + x4)))`
- **row 33509** | mu 320 -> 312 (ratio 0.975) | nodes 24 -> 25
  - orig: `* / - / x6 tanh + x15 x8 x9 sinh * * atanh <constant> - x9 tan x11 x15 abs atanh abs x3`
  - simp: `-abs(atanh(x3))*(x9 - x6/tanh(x15 + x8))/sinh(x15*atanh(<constant>)*(x9 + tan(-x11)))`
- **row 34112** | mu 211 -> 203 (ratio 0.962) | nodes 30 -> 30
  - orig: `+ - log asinh + x3 * * x2 + + + x2 x7 / rootn x17 5 5 rootn x8 3 + x13 x17 x8 tan * 5 x9`
  - simp: `-x8 + log(asinh(x3 + x2*(x13 + x17)*(x2 + x7 + 0.2*rootn(x17, 5) + rootn(x8, 3)))) - tan(-5*x9)`
- **row 35543** | mu 181 -> 173 (ratio 0.956) | nodes 26 -> 25
  - orig: `* + x2 / + * 4 x17 * 4 - x9 + atanh + abs x13 x2 asinh + x10 x10 x16 + x16 x9`
  - simp: `(x16 + x9)*(x2 + 4*(x17 + x9 + asinh(-2*x10) + atanh(-x2 - abs(x13)))/x16)`
- **row 51398** | mu 324 -> 316 (ratio 0.975) | nodes 27 -> 30
  - orig: `* / * + x15 rootn - x6 x12 2 x5 pow - x3 x16 3 / / log x4 + abs <constant> x17 / x11 x10`
  - simp: `-x10*x5*log(x4)*(x15 + (-x12 + x6)^(0.5))/(x11*(x16 - x3)^3*(x17 + <constant>))`
- **row 60531** | mu 951 -> 908 (ratio 0.955) | nodes 33 -> 29
  - orig: `+ * - asinh <constant> + * x13 <constant> + x10 + x10 * / * 5 x13 <constant> x13 + + + sinh <constant> / x10 <constant> x10 * x13 <constant> x13`
  - simp: `x13 - (x10 + <constant>*x10 + <constant>*x13 + <constant>)*(2*x10 + <constant>*x13 + <constant>*x13^2 + <constant>)`

## ratio == 1  (52167 rows, 79.60%)

- **row 3771** | mu 199 -> 199 (ratio 1.000) | nodes 26 -> 27
  - orig: `log neg inv + - / + x4 x8 tan / pow * 2 tan x12 2 x17 x14 / / * 3 x7 x4 x16`
  - simp: `log(-1/(-x14 + 3*x7/(x16*x4) + (x4 + x8)/tan(4*tan(x12)^2/x17)))`
- **row 12746** | mu 256 -> 256 (ratio 1.000) | nodes 32 -> 32
  - orig: `- cos x15 / tanh sin - x4 x17 / / / - x2 * + x1 x4 x15 x5 + * x2 x5 - x1 x17 + - x4 x2 x14`
  - simp: `cos(x15) - x5*tanh(sin(-x17 + x4))*(x1 - x17 + x2*x5)*(x14 - x2 + x4)/(x2 - x15*(x1 + x4))`
- **row 16854** | mu 194 -> 194 (ratio 1.000) | nodes 32 -> 29
  - orig: `rootn - / x3 rootn pow * * asinh - + * 4 x12 x12 x12 x3 - x3 - x3 x12 4 3 + x3 / x12 / x3 x12 2`
  - simp: `(-x3 + x3/rootn(x12^4*x3^4*asinh(4*x12)^4, 3) - x12^2/x3)^(0.5)`
- **row 18979** | mu 222 -> 222 (ratio 1.000) | nodes 34 -> 30
  - orig: `- / - x5 + x4 * 2 x15 x7 / - / - - x4 + - + x15 x1 rootn / x15 x15 4 + x7 cos x7 x15 x1 x15 x1`
  - simp: `(x15 + (x1 + 2*x15 - x4 + x7 + cos(x7) - 1)/x1)/x1 - (2*x15 + x4 - x5)/x7`
- **row 23673** | mu 213 -> 213 (ratio 1.000) | nodes 24 -> 26
  - orig: `/ * x8 x8 - / - * - x9 tanh x9 / x8 4 - x9 / x8 - x8 x9 x9 x8`
  - simp: `x8^2/(-x8 - (x9 - x8/(x8 - x9) - 0.25*x8*(x9 + tanh(-x9)))/x9)`
- **row 26792** | mu 182 -> 182 (ratio 1.000) | nodes 30 -> 28
  - orig: `* / pow - - cosh sinh x11 - x2 / / x16 4 5 x13 4 x13 * x4 - x5 * 5 * x16 log pow x4 5`
  - simp: `x4*(x13 - 0.05*x16 + x2 - cosh(sinh(x11)))^4*(x5 - 5*x16*log(x4^5))/x13`
- **row 26810** | mu 208 -> 208 (ratio 1.000) | nodes 30 -> 32
  - orig: `* x2 * / x4 / * x14 x7 x6 - / / * + x5 + exp x8 + x10 * x15 x16 tanh x17 atanh x10 x13 x3`
  - simp: `-x2*x4*x6*(x3 - tanh(x17)*(x10 + x5 + exp(x8) + x15*x16)/(x13*atanh(x10)))/(x14*x7)`
- **row 44930** | mu 277 -> 277 (ratio 1.000) | nodes 33 -> 35
  - orig: `/ + * x3 x13 * + / x15 3 * 5 x3 x6 / - sinh + cos / x14 x15 / x2 * / x5 x13 x9 + x2 x4 x16`
  - simp: `(x13*x3 + x6*(x15/3 + 5*x3))/(-(x2 + x4 + sinh(-cos(x14/x15) - x13*x2/(x5*x9)))/x16)`
- **row 45883** | mu 135 -> 135 (ratio 1.000) | nodes 27 -> 22
  - orig: `/ x7 + + / x7 * x7 - x7 / x7 x7 inv x7 + + acos x7 * rootn x7 5 + x7 x7 x7`
  - simp: `x7/(x7 + acos(x7) + 1/x7 + 1/(x7 - 1) + 2*x7*rootn(x7, 5))`
- **row 46441** | mu 120 -> 120 (ratio 1.000) | nodes 31 -> 17
  - orig: `- * + x9 x9 * 2 / / x9 + - + x3 acosh sinh x3 - x2 - + x2 x9 x3 x17 / x9 x9 / x2 5`
  - simp: `-0.2*x2 + 4*x9^2/(x17 + x9 + acosh(sinh(x3)))`
- **row 58990** | mu 257 -> 257 (ratio 1.000) | nodes 30 -> 30
  - orig: `/ + x2 x12 - / atan * atanh - x5 x11 x3 5 pow sin - x10 - + rootn - x8 * x1 x14 5 x7 x8 4`
  - simp: `(x12 + x2)/(0.2*atan(x3*atanh(-x11 + x5)) - sin(x10 - x7 + x8 - rootn(x8 - x1*x14, 5))^4)`
- **row 59103** | mu 289 -> 289 (ratio 1.000) | nodes 32 -> 28
  - orig: `- - pow / x3 4 2 - x8 + / + x4 * 5 cosh / * x4 asin x3 3 4 + + x3 * sin x8 <constant> x4 x4`
  - simp: `x3 + 0.25*x4 - x8 + 1.25*cosh(x4*asin(x3)/3) + 0.0625*x3^2 + <constant>*sin(x8)`

## ratio > 1  (0 rows, 0.00%)

