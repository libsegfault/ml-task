## Отчёт по практическому заданию на тему "Генерация имени функции по ее телу"

### Подзадача 1: Подготовка набора данных

Для подготовки данных из CodeSearchNet была использована библиотека tree-sitter (query).

Примеры извлечённых данных:
- Для функции
```py
def ucas_download_playlist(url, output_dir = '.', merge = False, info_only = False, **kwargs):
    '''course page'''
    html = get_content(url)

    parts = re.findall( r'(getplaytitle.do\?.+)"', html)
    assert parts, 'No part found!'

    for part_path in parts:
        ucas_download('http://v.ucas.ac.cn/course/' + part_path, output_dir=output_dir, merge=merge, info_only=info_only)

```
- Имя: `ucas_download_playlist`
- Тело:
```py
'''course page'''
    html = get_content(url)

    parts = re.findall( r'(getplaytitle.do\?.+)"', html)
    assert parts, 'No part found!'

    for part_path in parts:
        ucas_download('http://v.ucas.ac.cn/course/' + part_path, output_dir=output_dir, merge=merge, info_only=info_only)
```
- Тело без комментариев:
```py

    html = get_content(url)

    parts = re.findall( r'(getplaytitle.do\?.+)"', html)
    assert parts, 'No part found!'

    for part_path in parts:
        ucas_download('http://v.ucas.ac.cn/course/' + part_path, output_dir=output_dir, merge=merge, info_only=info_only)
```
- Для функции
```go
func (rs readSet) cmps() []v3.Cmp {
	cmps := make([]v3.Cmp, 0, len(rs))
	for k, rk := range rs {
		cmps = append(cmps, isKeyCurrent(k, rk))
	}
	return cmps
}
```
- Имя - `cmps`
- Тело:
```go
{
	cmps := make([]v3.Cmp, 0, len(rs))
	for k, rk := range rs {
		cmps = append(cmps, isKeyCurrent(k, rk))
	}
	return cmps
}

```
- Тело без комментариев совпадает с телом, так как комментариев в этой функции нет.

### Подзадача 2: Использование предобученных моделей для предсказания имен функций

Была выбрана модель CodeT5+.

Для `python` метрики получились следующие:

|  Метрика    |  Без комментариев |   С комментариями  |
| ----------- | ----------------- | ------------------ |
| exact_match |       0.1         |        0.22        |
| rouge1      |  0.3166800976801  | 0.4546626984126984 |
| rouge2      |  0.1239285714286  | 0.2390476190476191 |
| rougeL      |  0.3128348595849  | 0.4364246031746031 |
| rougeLsum   |  0.3153110500611  | 0.4363531746031744 |

А для `go`:

|  Метрика    |  Без комментариев |   С комментариями  |
| ----------- | ----------------- | ------------------ |
| exact_match |       0.031       |        0.026       |
| rouge1      |       0.057       |        0.054       |
| rouge2      |       0.0         |        0.0         |
| rougeL      |       0.057       |        0.054       |
| rougeLsum   |       0.056       |        0.054       |


Результаты ожидаемо плохие, например, для парсинга `go` лучше подходят его же собственные пакеты  (`go/ast`, `go/parser`, `go/printer`, `go/token` и т.д.),
хотя в целом это весьма типичный результат для техник машинного обучения, работающих в условиях отсутствия огромного датацентра, особенно если они "меряют погоду".
Для большинства функций существует больше одного "хорошего" имени, причём они могут даже не иметь ни одного общего слова.
