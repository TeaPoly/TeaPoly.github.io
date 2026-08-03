---
layout: page
title: Tech
description: Technology articles archive.
hide_title: true
permalink: /categories/tech/
---

<section class="plain-section list-section cave-articles">
    <div class="section-head section-head--compact cave-articles__head">
        <h3 class="section-head__title">Articles</h3>
        <div class="cave-years" role="tablist" aria-label="Article filters">
            <a class="cave-years__item" href="{{ '/all/' | prepend: site.baseurl }}" aria-selected="false">All</a>
            <span class="cave-years__sep" aria-hidden="true">·</span>
            <a class="cave-years__item is-active" href="{{ '/categories/tech/' | prepend: site.baseurl }}" aria-selected="true">Tech</a>
            <span class="cave-years__sep" aria-hidden="true">·</span>
            <a class="cave-years__item" href="{{ '/categories/life/' | prepend: site.baseurl }}" aria-selected="false">Life</a>
        </div>
    </div>

    <ul class="cave-posts" aria-label="Tech articles">
        {% for post in site.categories.tech %}
            <li class="cave-posts__item">
                <a class="cave-posts__link" href="{{ post.url | prepend: site.baseurl }}">
                    <span class="cave-posts__date">{{ post.date | date: "%m/%d" }}</span>
                    <span class="cave-posts__title">{{ post.title }}</span>
                    <span class="cave-posts__arrow" aria-hidden="true">›</span>
                </a>
            </li>
        {% endfor %}
    </ul>
</section>
