// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import io from 'socket.io-client'
import VueSocketIO from 'vue-socket.io'

Vue.config.productionTip = false

export const SocketInstance = io('http://localhost:5000')

Vue.use(new VueSocketIO({
  debug: true,
  connection: SocketInstance
}))

/* eslint-disable no-new */
new Vue({
  router,
  render: h => h(App),
  components: { App },
  template: '<App/>'
}).$mount('#app')
