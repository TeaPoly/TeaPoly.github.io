---
layout: page
title: Categories
description: Article categories.
hide_title: true
permalink: /categories/
---

<section class="plain-section list-section cave-articles">
    <ul class="cave-posts" aria-label="Category pages">
        <li class="cave-posts__item">
            <a class="cave-posts__link" href="{{ '/all/' | prepend: site.baseurl }}">
                <span class="cave-posts__date">All</span>
                <span class="cave-posts__title">All articles</span>
                <span class="cave-posts__arrow" aria-hidden="true">›</span>
            </a>
        </li>
        <li class="cave-posts__item">
            <a class="cave-posts__link" href="{{ '/categories/tech/' | prepend: site.baseurl }}">
                <span class="cave-posts__date">Tech</span>
                <span class="cave-posts__title">Technology articles</span>
                <span class="cave-posts__arrow" aria-hidden="true">›</span>
            </a>
        </li>
        <li class="cave-posts__item">
            <a class="cave-posts__link" href="{{ '/categories/life/' | prepend: site.baseurl }}">
                <span class="cave-posts__date">Life</span>
                <span class="cave-posts__title">Life articles</span>
                <span class="cave-posts__arrow" aria-hidden="true">›</span>
            </a>
        </li>
    </ul>
</section>
