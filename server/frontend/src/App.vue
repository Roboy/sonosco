<template>
  <div id="app">
    <div class="header">
      <img src="./assets/sonosco.png" style="margin: 10px" align="center">
    </div>
    <cookie-law>
      <div slot="message">
        This site uses cookies to match your transcripts.
      </div>
    </cookie-law>
    <router-view class="main"/>

    <md-speed-dial :class="bottomPosition">
      <md-speed-dial-target @click="addModel">
        <md-icon>add</md-icon>
      </md-speed-dial-target>
    </md-speed-dial>

  </div>
</template>

<script>
  import CookieLaw from 'vue-cookie-law'

export default {
  name: 'App',
  data: () => ({
    bottomPosition: 'md-bottom-left'
  }),

  methods: {
    addModel () {
      console.log('Add model')
      this.$router.push('/add')
    }
  },

  components: { CookieLaw }
}
</script>

<style>
  @import '//fonts.googleapis.com/css?family=Roboto:300,400,500,700,400italic|Material+Icons';

  html, body {
    min-height: 100vh;
    overflow: auto;
  }

  #app {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    /*text-align: center;*/
    color: #2c3e50;
    min-height: 100vh;
    display: flex;
    flex: 1;
    flex-direction: column;
  }

  .main {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .header {
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    /*justify-items: stretch;*/
    /*flex: 0 0 100%;*/
  }

  .header h1 {
    justify-content: center;
  }

  .header img {
    position: absolute;
    top: 1px;
    right: 10px;
  }
</style>
