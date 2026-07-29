---
layout: page
title: Cave
description: Articles, hobbies, and notes gathered in one place.
hide_title: true
permalink: /cave/
---

<section class="plain-section list-section cave-articles">
    <div class="section-head section-head--compact cave-articles__head">
        <h3 class="section-head__title">Articles</h3>
        <div class="cave-years" role="tablist" aria-label="Article years">
            <a class="cave-years__item is-active" href="{{ '/cave/' | prepend: site.baseurl }}" aria-selected="true">All</a>
            <span class="cave-years__sep" aria-hidden="true">·</span>
            <a class="cave-years__item" href="{{ '/categories/tech/' | prepend: site.baseurl }}" aria-selected="false">Tech</a>
            <span class="cave-years__sep" aria-hidden="true">·</span>
            <a class="cave-years__item" href="{{ '/categories/life/' | prepend: site.baseurl }}" aria-selected="false">Life</a>
        </div>
    </div>

    <ul class="cave-posts" aria-label="Articles">
        {% for post in site.posts limit: 4 %}
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

<section class="plain-section cave-hobbies">
    <div class="section-head section-head--compact">
        <h3 class="section-head__title">Hobbies</h3>
        <span class="section-head__meta">Books · music · films · fashion</span>
    </div>

    <div class="hobby-bento" aria-label="Hobbies">
        <article class="hobby-panel hobby-panel--books">
            <div class="section-head section-head--compact">
                <h3 class="section-head__title">Books</h3>
                <span class="section-head__meta">6 books</span>
            </div>

            <div class="item-shelf" role="list">
                <div class="item-shelf__row" role="listitem">
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind" target="_blank" rel="noopener noreferrer" title="Sapiens: A Brief History of Humankind">
                        <img src="{{ '/assets/images/hobbies/sapiens.svg' | prepend: site.baseurl }}" alt="Sapiens: A Brief History of Humankind" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/The_Courage_to_Be_Disliked" target="_blank" rel="noopener noreferrer" title="The Courage to Be Disliked">
                        <img src="{{ '/assets/images/hobbies/courage-to-be-disliked.svg' | prepend: site.baseurl }}" alt="The Courage to Be Disliked" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Flow_(psychology)" target="_blank" rel="noopener noreferrer" title="Flow">
                        <img src="{{ '/assets/images/hobbies/flow.svg' | prepend: site.baseurl }}" alt="Flow" width="120" height="180" loading="lazy">
                    </a>
                </div>
                <div class="item-shelf__row" role="listitem">
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow" target="_blank" rel="noopener noreferrer" title="Thinking, Fast and Slow">
                        <img src="{{ '/assets/images/hobbies/thinking-fast-and-slow.svg' | prepend: site.baseurl }}" alt="Thinking, Fast and Slow" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Poor_Charlie%27s_Almanack" target="_blank" rel="noopener noreferrer" title="Poor Charlie's Almanack">
                        <img src="{{ '/assets/images/hobbies/poor-charlies-almanack.svg' | prepend: site.baseurl }}" alt="Poor Charlie's Almanack" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/The_Selfish_Gene" target="_blank" rel="noopener noreferrer" title="The Selfish Gene">
                        <img src="{{ '/assets/images/hobbies/selfish-gene.svg' | prepend: site.baseurl }}" alt="The Selfish Gene" width="120" height="180" loading="lazy">
                    </a>
                </div>
            </div>
        </article>

        <article class="hobby-panel hobby-panel--music">
            <div class="section-head section-head--compact">
                <h3 class="section-head__title">Music</h3>
                <span class="section-head__meta">3 picks</span>
            </div>

            <div class="music-stack">
                <div class="music-stack__tooltip">
                    <span class="music-stack__artist">Michael Jackson</span>
                    <strong class="music-stack__track">Thriller</strong>
                </div>

                <div class="music-stack__covers">
                    <a class="music-stack__cover is-active" href="https://en.wikipedia.org/wiki/Thriller_(album)" target="_blank" rel="noopener noreferrer" title="Michael Jackson — Thriller">
                        <img src="{{ '/assets/images/hobbies/thriller.svg' | prepend: site.baseurl }}" alt="Thriller" width="148" height="148" loading="lazy">
                    </a>
                    <a class="music-stack__cover" href="https://en.wikipedia.org/wiki/Sgt._Pepper%27s_Lonely_Hearts_Club_Band" target="_blank" rel="noopener noreferrer" title="The Beatles — Sgt. Pepper's Lonely Hearts Club Band">
                        <img src="{{ '/assets/images/hobbies/sgt-pepper.svg' | prepend: site.baseurl }}" alt="Sgt. Pepper's Lonely Hearts Club Band" width="148" height="148" loading="lazy">
                    </a>
                    <a class="music-stack__cover" href="https://www.youtube.com/results?search_query=You+Are+My+Sunshine" target="_blank" rel="noopener noreferrer" title="Traditional — You Are My Sunshine">
                        <img src="{{ '/assets/images/hobbies/you-are-my-sunshine.svg' | prepend: site.baseurl }}" alt="You Are My Sunshine" width="148" height="148" loading="lazy">
                    </a>
                </div>
            </div>
        </article>

        <article class="hobby-panel hobby-panel--movies">
            <div class="section-head section-head--compact">
                <h3 class="section-head__title">Movies</h3>
                <span class="section-head__meta">9 films</span>
            </div>

            <div class="item-shelf" role="list">
                <div class="item-shelf__row" role="listitem">
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/The_Godfather" target="_blank" rel="noopener noreferrer" title="The Godfather">
                        <img src="{{ '/assets/images/hobbies/godfather.svg' | prepend: site.baseurl }}" alt="The Godfather" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/The_Shawshank_Redemption" target="_blank" rel="noopener noreferrer" title="The Shawshank Redemption">
                        <img src="{{ '/assets/images/hobbies/shawshank.svg' | prepend: site.baseurl }}" alt="The Shawshank Redemption" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Schindler%27s_List" target="_blank" rel="noopener noreferrer" title="Schindler's List">
                        <img src="{{ '/assets/images/hobbies/schindlers-list.svg' | prepend: site.baseurl }}" alt="Schindler's List" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Saving_Private_Ryan" target="_blank" rel="noopener noreferrer" title="Saving Private Ryan">
                        <img src="{{ '/assets/images/hobbies/saving-private-ryan.svg' | prepend: site.baseurl }}" alt="Saving Private Ryan" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Inception" target="_blank" rel="noopener noreferrer" title="Inception">
                        <img src="{{ '/assets/images/hobbies/inception.svg' | prepend: site.baseurl }}" alt="Inception" width="120" height="180" loading="lazy">
                    </a>
                </div>
                <div class="item-shelf__row" role="listitem">
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Fight_Club" target="_blank" rel="noopener noreferrer" title="Fight Club">
                        <img src="{{ '/assets/images/hobbies/fight-club.svg' | prepend: site.baseurl }}" alt="Fight Club" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Blade_Runner_2049" target="_blank" rel="noopener noreferrer" title="Blade Runner 2049">
                        <img src="{{ '/assets/images/hobbies/blade-runner-2049.svg' | prepend: site.baseurl }}" alt="Blade Runner 2049" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/Oppenheimer_(film)" target="_blank" rel="noopener noreferrer" title="Oppenheimer">
                        <img src="{{ '/assets/images/hobbies/oppenheimer.svg' | prepend: site.baseurl }}" alt="Oppenheimer" width="120" height="180" loading="lazy">
                    </a>
                    <a class="item-shelf__item" href="https://en.wikipedia.org/wiki/The_Great_Buddha%2B" target="_blank" rel="noopener noreferrer" title="The Great Buddha +">
                        <img src="{{ '/assets/images/hobbies/great-buddha.svg' | prepend: site.baseurl }}" alt="The Great Buddha +" width="120" height="180" loading="lazy">
                    </a>
                </div>
            </div>
        </article>

        <article class="hobby-panel hobby-panel--fashion">
            <div class="section-head section-head--compact">
                <h3 class="section-head__title">Fashion</h3>
                <span class="section-head__meta">5 labels</span>
            </div>

            <div class="item-shelf item-shelf--labels" role="list">
                <div class="item-shelf__row item-shelf__row--compact item-shelf__row--brands" role="listitem">
                    <a class="fashion-inline fashion-inline--logo" href="https://www.jjjjound.com/" target="_blank" rel="noopener noreferrer" aria-label="JJJJound">
                        <img src="{{ '/assets/images/hobbies/jjjjound.svg' | prepend: site.baseurl }}" alt="JJJJound" width="80" height="80" loading="lazy">
                    </a>
                    <a class="fashion-inline fashion-inline--logo" href="https://mdnsonline.com/" target="_blank" rel="noopener noreferrer" aria-label="MADNESS">
                        <img src="{{ '/assets/images/hobbies/madness.svg' | prepend: site.baseurl }}" alt="MADNESS" width="80" height="80" loading="lazy">
                    </a>
                    <a class="fashion-inline fashion-inline--logo" href="https://www.vans.com/" target="_blank" rel="noopener noreferrer" aria-label="Vans">
                        <img src="{{ '/assets/images/hobbies/vans.svg' | prepend: site.baseurl }}" alt="Vans" width="80" height="80" loading="lazy">
                    </a>
                    <a class="fashion-inline fashion-inline--logo" href="https://www.converse.com/" target="_blank" rel="noopener noreferrer" aria-label="Converse">
                        <img src="{{ '/assets/images/hobbies/converse.svg' | prepend: site.baseurl }}" alt="Converse" width="80" height="80" loading="lazy">
                    </a>
                    <a class="fashion-inline fashion-inline--logo" href="https://retawstyle.com/" target="_blank" rel="noopener noreferrer" aria-label="retaW">
                        <img src="{{ '/assets/images/hobbies/retaw.svg' | prepend: site.baseurl }}" alt="retaW" width="80" height="80" loading="lazy">
                    </a>
                </div>
            </div>
        </article>
    </div>
</section>


