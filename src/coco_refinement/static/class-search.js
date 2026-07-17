function editDistance(a, b) {
  const previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    let diagonal = previous[0];
    previous[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const old = previous[j];
      previous[j] = a[i - 1] === b[j - 1]
        ? diagonal
        : Math.min(diagonal + 1, previous[j] + 1, previous[j - 1] + 1);
      diagonal = old;
    }
  }
  return previous[b.length];
}

function normalizeSearchText(value) {
  return String(value ?? '').toLocaleLowerCase('en-US').trim().split(/\s+/u).filter(Boolean).join(' ');
}

function commonPrefixLength(left, right) {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) index += 1;
  return index;
}

function candidateSpellingScore(query, candidate) {
  if (!query) return { distance: candidate.length, prefix: 0 };
  const queryWords = query.split(' ');
  const candidateWords = candidate.split(' ');
  const comparisons = [candidate];
  if (queryWords.length <= candidateWords.length) {
    for (let index = 0; index <= candidateWords.length - queryWords.length; index += 1) {
      comparisons.push(candidateWords.slice(index, index + queryWords.length).join(' '));
    }
  }
  return comparisons.map(value => ({
    distance: editDistance(query, value),
    prefix: commonPrefixLength(query, value),
  })).sort((left, right) => left.distance - right.distance || right.prefix - left.prefix)[0];
}

function spellingThreshold(query) {
  const compactLength = query.replaceAll(' ', '').length;
  if (!compactLength) return 0;
  return Math.min(3, Math.max(1, Math.ceil(compactLength / 3)));
}

export function rankCategories(categories, query) {
  const needle = normalizeSearchText(query);
  if (!needle) return categories.slice();
  return categories.map((category, index) => {
    const name = normalizeSearchText(category.name);
    let rank = 4;
    let distance = 99;
    let prefix = 0;
    if (name === needle) rank = 0;
    else if (name.startsWith(needle)) rank = 1;
    else if (name.includes(needle)) rank = 2;
    else {
      ({ distance, prefix } = candidateSpellingScore(needle, name));
      if (distance <= spellingThreshold(needle)) rank = 3;
    }
    return { category, rank, distance, prefix, index };
  }).filter(item => item.rank < 4)
    .sort((a, b) => a.rank - b.rank || a.distance - b.distance || b.prefix - a.prefix || a.index - b.index)
    .map(item => item.category);
}

export function installCategorySearch({
  input,
  results,
  selection,
  categories,
  onChoose = () => {},
  onClear = () => {},
}) {
  if (typeof onChoose !== 'function' || typeof onClear !== 'function') {
    throw new TypeError('category callbacks must be functions');
  }
  let active = -1;
  let current = [];
  let selected = null;
  const clearSelection = ({ notify = true } = {}) => {
    const prior = selected;
    selected = null;
    delete selection.dataset.categoryId;
    delete selection.dataset.categoryName;
    selection.textContent = 'No category selected.';
    if (notify && prior) onClear(prior);
  };
  const choose = (category, { notify = true } = {}) => {
    selected = category;
    selection.textContent = `Selected: ${category.name} (COCO id ${category.id})`;
    selection.dataset.categoryId = String(category.id);
    selection.dataset.categoryName = category.name;
    input.value = category.name;
    results.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    if (notify) onChoose({ id: category.id, name: category.name });
  };
  const render = () => {
    current = rankCategories(categories, input.value).slice(0, 12);
    active = current.length ? 0 : -1;
    results.replaceChildren(...current.map((category, index) => {
      const item = document.createElement('li');
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', String(index === active));
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = category.name;
      const id = document.createElement('span');
      id.className = 'category-id';
      id.textContent = String(category.id);
      button.append(id);
      button.addEventListener('click', () => choose(category));
      item.append(button);
      return item;
    }));
    results.hidden = !current.length;
    input.setAttribute('aria-expanded', String(Boolean(current.length)));
  };
  input.addEventListener('input', () => {
    if (!selected || normalizeSearchText(input.value) !== normalizeSearchText(selected.name)) clearSelection();
    render();
  });
  input.addEventListener('focus', render);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowDown' && current.length) { event.preventDefault(); active = (active + 1) % current.length; renderActive(); }
    if (event.key === 'ArrowUp' && current.length) { event.preventDefault(); active = (active - 1 + current.length) % current.length; renderActive(); }
    if (event.key === 'Enter' && active >= 0 && current[active]) { event.preventDefault(); choose(current[active]); }
    if (event.key === 'Escape') { results.hidden = true; input.setAttribute('aria-expanded', 'false'); }
  });
  const renderActive = () => [...results.children].forEach((item, index) => item.setAttribute('aria-selected', String(index === active)));
  return { choose, clearSelection, getSelection: () => selected, refresh: render };
}
