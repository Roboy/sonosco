import Vue from 'vue'
import Router from 'vue-router'
import Transcriptor from '@/components/Transcriptor'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Transcriptor',
      component: Transcriptor
    }
  ]
})
