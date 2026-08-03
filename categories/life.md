---
layout: page
title: Life
description: Life articles archive.
hide_title: true
permalink: /categories/life/
---

<section class="plain-section list-section cave-articles">
    <div class="section-head section-head--compact cave-articles__head">
        <h3 class="section-head__title">Articles</h3>
        <div class="cave-years" role="tablist" aria-label="Article filters">
            <a class="cave-years__item" href="{{ '/all/' | prepend: site.baseurl }}" aria-selected="false">All</a>
            <span class="cave-years__sep" aria-hidden="true">·</span>
            <a class="cave-years__item" href="{{ '/categories/tech/' | prepend: site.baseurl }}" aria-selected="false">Tech</a>
            <span class="cave-years__sep" aria-hidden="true">·</span>
            <a class="cave-years__item is-active" href="{{ '/categories/life/' | prepend: site.baseurl }}" aria-selected="true">Life</a>
        </div>
    </div>

    <ul class="cave-posts" aria-label="Life articles">
        {% for post in site.categories.life %}
            <li class="cave-posts__item">
                <a class="cave-posts__link" href="{{ post.url | prepend: site.baseurl }}">
                    <span class="cave-posts__date">{{ post.date | date: "%Y/%m/%d" }}</span>
                    <span class="cave-posts__title">{{ post.title }}</span>
                    <span class="cave-posts__arrow" aria-hidden="true">›</span>
                </a>
            </li>
        {% endfor %}
    </ul>
</section>
